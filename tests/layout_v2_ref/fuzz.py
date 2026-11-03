"""Seeded fuzz documents for the layout v2 differential test (Python reference vs Rust port), K5-A1.

`fuzz_documents(seed, n_docs)` yields documents: `{"name", "kind", "pages": [(page_no, CharTable, plain)],
"context": [(page_no, CharTable)]}`. Every table is round-tripped through `k5chars.dumps/loads`, so both sides read
the same f32-quantised values. Kinds:
  synth      synthetic pages: paragraphs (indents, hanging indents, short last lines), headings (big, bold,
             numbered, emphasis), lists (bullets, enumerations, PUA bullets), captions (Greek/Latin, lookalike
             letters), tables (cell gaps, numeric, row mates), column-ordered cells, formulas (math fonts, operators),
             chart text, footnotes, running headers/footers and page numbers repeated over the document, rotated
             content and margin lines, rows joined without a newline, superscripts, mixed scripts (Greek NFC and
             decomposed, Latin, Cyrillic, other-script digits, superscript digits, fractions, PUA, combining marks,
             supplementary chars as surrogate pairs, lone surrogates, Python-only whitespace such as U+001C..1F)
  perturbed  real tuning/context tables (given by the caller) with jittered/scaled boxes, quarter-turn angle
             changes, font renames, dropped newlines, truncation, degenerate boxes
  edge       empty table, one char, only spaces/newlines, plain without table, table without plain
Each page's plain text is the table's own text (the production case) or, for ~15% of pages, a perturbed copy (the
difflib alignment path) or "".
"""
from __future__ import annotations

import math
import random
import struct

from . import k5chars as kc
from . import labels_v2 as lv

F32 = struct.Struct("<f")

GREEK = "αβγδεζηθικλμνξοπρσςτυφχψωάέήίόύώϊϋΐΰΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΆΈΉΊΌΎΏ"
GREEK_DECOMP = ["ά", "έ", "ΐ", "ῷ", "Ά", "ϋ"]
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZéèçñüößſıİK"
CYR = "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗ"
DIGITS = "0123456789"
ODD_DIGITS = "٠١٢٣४५६²³¹½¾ⅣⅫ①"
PUNCT = ".,;:!?()[]-–—·'\"«»/%*†‡"
ODD_SPACE = [" ", " ", "　", "\u001c", "\u001f", "\u0085", " ", "\t"]
SUPP = ["\U0001d400", "\U0001f600", "\U00010330"]
PUA = ["", "", "", ""]
BULLETS = ["•", "–", "▪", "■", "◦", "●", "➢", "►", "✓", "·", "-", "*", "", "o"]
ENUMS = ["1.", "2)", "(3)", "a.", "(b)", "α)", "β.", "iv.", "(ii)", "Α.", "Β)", "12 α .", "5α)", "o ", "IV."]
NUMBERED = ["1.", "2.1.", "3.3.2.", "IV.", "ΙV.1.1.", "Α.1", "Β.2.3", "ΚΕΦΑΛΑΙΟ 2", "Κεφάλαιο 3", "Chapter 4",
            "Appendix A", "ΠΑΡΑΡΤΗΜΑ", "Μέρος Β", "10.", "7"]
CAPTIONS = ["Πίνακας 3:", "ΠΙΝΑΚΑΣ 2", "Σχήμα 2.1", "Σχ. 4", "Εικόνα 5", "Εικ. 1", "Διάγραμμα 3", "∆ιάγραµµα 1",
            "Γράφημα 2", "Χάρτης 1", "Φωτογραφία 7", "Figure 4", "Fig. 2", "Table 1", "TABLE 3", "Chart 2",
            "Graph 1", "Map 5", "Photo 9", "Πίν. 4", "Πιν. 2", "FIGURE 6"]
FONTS = ["ABCDEF+TimesNewRoman", "Times-Roman", "ArialMT", "Calibri", "ABCDEF+Times-Bold", "Arial,Bold",
         "Calibri-Bold", "Helvetica-BoldOblique", "Calibri,Italic", "Times-Italic", "GRXTimes", "CMR10", "CMBX12",
         "CMMI10", "CMSY8", "Symbol", "MTExtra", "MT Extra", "Cambria Math", "Euclid", "Garamond-Semibold",
         "Myriad-Black", "Heavy", "Font-B", "Font,B", "Font-Bd", "", "ſemibold", "BOLD\n", "x+y+DEMI",
         "KHeavy", "İtalic", "font-IT", "BX9", "ϐold"]


def f32(v):
    return F32.unpack(F32.pack(v))[0]


def roundtrip(t):
    return kc.loads(kc.dumps(t))


def words(rng, script, n):
    out = []
    for _ in range(n):
        k = rng.randint(1, 10)
        r = rng.random()
        if script == "el":
            w = "".join(rng.choice(GREEK) for _ in range(k))
            if r < 0.08:
                w += rng.choice(GREEK_DECOMP)
        elif script == "la":
            w = "".join(rng.choice(LATIN) for _ in range(k))
        elif script == "ru":
            w = "".join(rng.choice(CYR) for _ in range(k))
        else:
            w = "".join(rng.choice(GREEK + LATIN + CYR + DIGITS) for _ in range(k))
        if r < 0.05:
            w += rng.choice(PUNCT)
        elif r < 0.08:
            w = rng.choice(ODD_DIGITS) + w
        elif r < 0.09:
            w += rng.choice(SUPP)
        elif r < 0.10:
            w += rng.choice(PUA)
        elif r < 0.11:
            w = "".join(rng.choice(DIGITS) for _ in range(rng.randint(1, 5)))
        out.append(w)
    sep = " "
    if rng.random() < 0.05:
        sep = rng.choice(ODD_SPACE)
    return sep.join(out)


class Page:
    """Builds a k5chars table: rows are laid out in the reading frame of their angle and mapped to user space."""

    def __init__(self, rng, w, h, rotation, page_no):
        self.rng = rng
        self.t = kc.CharTable(w, h, rotation, page_no)
        self.fonts = {}
        self.w, self.h = w, h

    def font(self, name):
        return self.fonts.setdefault(name, len(self.fonts))

    def _user_box(self, angle, x, top, wd, size):
        """reading-frame box (x .. x+wd along the baseline, top .. top+size down) -> PDF user-space (l, b, r, t)."""
        w, h = self.w, self.h
        if angle == 0:
            return (x, h - top - size, x + wd, h - top)
        if angle == 270:       # text running upwards: baseline along +y, glyph up towards -x
            return (top, x, top + size, x + wd)
        if angle == 180:
            return (w - x - wd, top, w - x, top + size)
        return (w - top - size, h - x - wd, w - top, h - x)    # 90: running downwards

    def add(self, ch, box, loose, size, weight, angle_rad, font, flags):
        t = self.t
        t.ch.append(ch)
        t.bbox.append(tuple(box))
        t.loose.append(tuple(loose))
        t.size.append(size)
        t.weight.append(weight)
        t.angle.append(angle_rad)
        t.font.append(font)
        t.flags.append(flags)

    def row(self, text, x, top, size, font="Times-Roman", angle=0, weight=400, flags=0, jitter=0.0, advance=0.5,
            loose_mode=None):
        rng = self.rng
        fid = self.font(font)
        cx = x
        rad = math.radians(angle)
        for c in text:
            units = [c]
            if ord(c) > 0xFFFF and rng.random() < 0.8:          # UTF-16 surrogate pair, one index per unit
                v = ord(c) - 0x10000
                units = [chr(0xD800 + (v >> 10)), chr(0xDC00 + (v & 0x3FF))]
            wd = advance * size * (0.6 if c in "il.,;:'" else 1.0)
            for u in units:
                jt = rng.uniform(-jitter, jitter) if jitter else 0.0
                box = self._user_box(angle, cx, top + jt, wd, size * 0.7)
                lb = self._user_box(angle, cx, top - 0.15 * size + jt, wd, size * 1.2)
                mode = loose_mode or ("normal" if rng.random() > 0.01 else rng.choice(["zero", "nan", "huge", "flat"]))
                if mode == "zero":
                    lb = (lb[0], lb[1], lb[0], lb[1])
                elif mode == "nan":
                    box = (float("nan"),) * 4
                elif mode == "huge":
                    lb = (2e6, 0.0, 2e6 + 1, 1.0)
                elif mode == "flat":
                    box = (box[0], box[1], box[2], box[1])
                    lb = (lb[0], lb[1], lb[2], lb[1])
                if c.isspace() and rng.random() < 0.5:
                    box = lb = (0.0, 0.0, 0.0, 0.0)
                a = rad + (rng.uniform(-1e-4, 1e-4) if rng.random() < 0.02 else 0.0)
                self.add(u, box, lb, size, weight, a, fid, flags)
            cx += wd

    def newline(self, kind="\r\n"):
        for c in kind:
            self.add(c, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.0, 0, 0.0, 0, lv.GENERATED)

    def table(self):
        self.t.fonts = [f for f, _ in sorted(self.fonts.items(), key=lambda kv: kv[1])]
        return roundtrip(self.t)


def synth_document(rng, name):
    w, h = rng.choice([(595.0, 842.0), (842.0, 595.0), (612.0, 792.0), (rng.uniform(200, 1500), rng.uniform(200, 1500))])
    rotation = rng.choice([0] * 10 + [90, 180, 270] + ([45, 360] if rng.random() < 0.05 else []))
    content_angle = rng.choice([0] * 12 + [90, 180, 270])
    script = rng.choice(["el", "el", "la", "ru", "mix"])
    body = rng.choice([9.0, 10.0, 10.5, 11.0, 12.0])
    body_font = rng.choice(FONTS[:4])
    bold_font = rng.choice(FONTS[4:12])
    running = words(rng, script, rng.randint(2, 6))
    start = rng.choice([1, 1, 2, rng.randint(3, 300)])
    pages = []
    for k in range(rng.randint(2, 9)):
        pno = start + k
        pg = Page(rng, w, h, rotation, pno)
        synth_page(rng, pg, script, body, body_font, bold_font, running, pno, content_angle)
        pages.append((pno, pg.table()))
    return pages


def synth_page(rng, pg, script, body, body_font, bold_font, running, pno, angle):
    fw = pg.w if angle in (0, 180) else pg.h       # frame width / height of the content angle
    fh = pg.h if angle in (0, 180) else pg.w
    left, right = rng.uniform(40, 90), fw - rng.uniform(40, 90)
    top = rng.uniform(20, 60)
    lead = body * rng.uniform(1.1, 1.4)
    first = True

    def nl():
        nonlocal first
        if not first:
            r = rng.random()
            pg.newline("\r\n" if r < 0.85 else ("\n" if r < 0.9 else ("\r" if r < 0.93 else " ")))
        first = False

    def line(text, x, size=body, font=body_font, **kw):
        nl()
        pg.row(text, x, cur[0], size, font, angle=kw.pop("angle", angle), **kw)

    def chars_for(width, size):
        return max(1, int(width / (0.5 * size)))

    cur = [top]
    if rng.random() < 0.7:                       # running header (repeated) + page number
        line(running, left + rng.uniform(0, 5), body * 0.8)
        if rng.random() < 0.5:
            pg.row("   " + rng.choice([str(pno), lv_roman(pno), "- %d -" % pno, "[%d]" % pno, greek_num(pno)]),
                   right - 30, cur[0], body * 0.8, body_font, angle=angle)
        cur[0] += lead * 2
    jitter = 0.3 if rng.random() < 0.2 else 0.0
    while cur[0] < fh * 0.8:
        r = rng.random()
        if r < 0.32:                             # paragraph
            n = rng.randint(1, 7)
            indent = rng.choice([0.0, 0.0, 2 * body, -2 * body, 3 * body])
            for i in range(n):
                x = left + (indent if i == 0 and indent > 0 else (0 if indent >= 0 or i == 0 else -indent))
                width = (right - x) * (rng.uniform(0.3, 0.9) if i == n - 1 else rng.uniform(0.95, 1.0))
                txt = words(rng, script, 40)[:chars_for(width, body)].strip() or "x"
                if rng.random() < 0.05:          # superscript note number inside the line
                    nl()
                    pg.row(txt[: len(txt) // 2], x, cur[0], body, body_font, angle=angle, jitter=jitter)
                    pg.row(str(rng.randint(1, 99)), x + len(txt) // 2 * 0.5 * body, cur[0] - 0.3 * body, body * 0.6,
                           body_font, angle=angle)
                    pg.row(txt[len(txt) // 2:], x + (len(txt) // 2 + 2) * 0.5 * body, cur[0], body, body_font,
                           angle=angle)
                    first = False
                else:
                    line(txt, x, jitter=jitter)
                if rng.random() < 0.05:          # the next row is joined without a newline
                    pg.row(" ", right, cur[0], body, body_font, angle=angle)
                    first = True
                cur[0] += lead
            cur[0] += lead * rng.choice([0.0, 0.0, 0.5, 1.0])
        elif r < 0.45:                           # heading
            kind = rng.random()
            size = body * rng.choice([1.0, 1.0, 1.2, 1.5, 2.0])
            font = bold_font if kind < 0.6 else body_font
            txt = (rng.choice(NUMBERED) + " " if rng.random() < 0.5 else "") + words(rng, script, rng.randint(1, 6))
            if rng.random() < 0.2:
                txt += rng.choice([".", ":", ",", "..", ""])
            cur[0] += lead * rng.choice([0.5, 1.0, 1.5])
            flags = lv.ITALIC if rng.random() < 0.2 else 0
            weight = rng.choice([400, 700, 690, 650, 0, -5, 900])
            line(txt, left, size, font, flags=flags, weight=weight)
            for _ in range(rng.choice([0, 0, 1, 2, 3])):   # multi-row heading: rows short of the text edge, numbers
                cur[0] += size * rng.uniform(1.1, 1.3)
                extra = (rng.choice(NUMBERED) + " " if rng.random() < 0.3 else "") + words(rng, script, rng.randint(1, 9))
                line(extra[:chars_for(right - left, size)], left + rng.choice([0, 0, 0.5 * body]), size, font,
                     flags=flags, weight=weight)
            cur[0] += lead * rng.choice([1.0, 1.5, 2.0])
        elif r < 0.55:                           # list
            for _ in range(rng.randint(1, 5)):
                mark = rng.choice(BULLETS + ENUMS)
                txt = mark + rng.choice([" ", "  ", ""]) + words(rng, script, rng.randint(2, 12))
                line(txt[:chars_for(right - left, body)], left + rng.choice([0, 0, body]))
                cur[0] += lead
                if rng.random() < 0.4:
                    line(words(rng, script, 8)[:chars_for(right - left, body)], left + rng.choice([1.5, 2, 4]) * body)
                    cur[0] += lead
        elif r < 0.63:                           # table: cells with gaps, maybe numeric, maybe row mates
            ncol = rng.randint(2, 5)
            tfont = rng.choice([body_font, body_font, "CMMI10", "CMSY8"])
            for _ in range(rng.randint(2, 6)):
                cells = [rng.choice([words(rng, script, 1), str(rng.randint(0, 9999)), "%.1f" % rng.uniform(0, 99)])
                         for _ in range(ncol)]
                if rng.random() < 0.3:           # each cell its own plain line, same visual row (row mates)
                    x = left
                    for c in cells:
                        line(c, x, font=tfont)
                        x += (right - left) / ncol
                else:
                    line(("    " * rng.randint(1, 3)).join(cells), left, font=tfont)
                cur[0] += lead
        elif r < 0.67:                           # column-ordered cells (one cell per plain line, down each column)
            cols, rows_ = rng.randint(2, 3), rng.randint(3, 5)
            y0 = cur[0]
            for ci in range(cols):
                cur[0] = y0
                for _ in range(rows_):
                    line(rng.choice([words(rng, script, 1), str(rng.randint(1, 500))]), left + ci * 0.2 * (right - left))
                    cur[0] += lead
        elif r < 0.73:                           # caption, maybe with chart text around it
            if rng.random() < 0.5:
                for _ in range(rng.randint(1, 4)):
                    line(rng.choice([str(rng.randint(0, 100)), words(rng, script, 1), "0 - 20"]),
                         left + rng.uniform(0, 200), body * rng.choice([0.7, 1.0]))
                    cur[0] += lead * rng.choice([1.0, -0.5, 2.0])
            cap = rng.choice(CAPTIONS) + " " + words(rng, script, rng.randint(1, 8))
            line(cap[:chars_for(right - left, body)], left + rng.choice([0, 40]), body * rng.choice([0.9, 1.0]),
                 rng.choice([body_font, bold_font]))
            cur[0] += lead * 1.5
            if rng.random() < 0.5:
                line(words(rng, script, 6)[:chars_for(right - left, body)], left)
                cur[0] += lead
        elif r < 0.78:                           # formula
            ops = "=+−×÷±∑√≤≥≈≠∂∈→∫"
            txt = " ".join(rng.choice(["x", "y", "α", "β", "2", "i", "n", rng.choice(ops), "(", ")", "^", "_"])
                           for _ in range(rng.randint(3, 15)))
            line(txt, left + rng.uniform(20, 120), body, rng.choice(["CMMI10", "CMSY8", body_font, "Symbol"]))
            cur[0] += lead * 1.5
        elif r < 0.82:                           # rotated margin line / rotated block inside the page
            other = rng.choice([0, 90, 180, 270])
            line(words(rng, script, rng.randint(1, 8)), rng.uniform(50, 300), body, angle=other)
            cur[0] += lead
        else:
            cur[0] += lead
    if rng.random() < 0.6:                       # footnotes in the lower part
        cur[0] = max(cur[0], fh * 0.7)
        for _ in range(rng.randint(1, 3)):
            mark = rng.choice(["1", "2", "12", "¹", "²", "*", "†", "9"])
            if rng.random() < 0.2:
                line(mark, left, body * 0.75)
                cur[0] += lead * 0.8
                line(words(rng, script, 10), left, body * 0.75)
            else:
                line(mark + " " + words(rng, script, 10)[:chars_for(right - left, body * 0.75)], left, body * 0.75)
            cur[0] += lead * 0.8
    if rng.random() < 0.7:                       # page footer: page number or repeated line
        cur[0] = fh - rng.uniform(15, 40)
        line(rng.choice([str(pno), "- %d -" % pno, lv_roman(pno), greek_num(pno), running, "Σελίδα %d" % pno]),
             fw / 2 - 10, body * 0.9)


def lv_roman(n):
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
            (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out or "0"


def greek_num(n):
    return "αβγδεϛζηθ"[n % 9] + "΄"


def perturb_tables(rng, tables):
    """Apply one perturbation recipe to every table of a real document (page, CharTable) list."""
    recipe = rng.choice(["jitter", "scale", "angle", "fonts", "nonewline", "truncate", "degenerate", "rotation",
                         "mixed"])
    out = []
    for pno, t in tables:
        t = kc.loads(kc.dumps(t))
        n = len(t.ch)
        steps = [recipe] if recipe != "mixed" else rng.sample(["jitter", "scale", "angle", "fonts", "nonewline",
                                                                "truncate", "degenerate", "rotation"], 3)
        for st in steps:
            n = len(t.ch)
            if st == "jitter":
                a = rng.choice([0.05, 0.5, 2.0])
                t.bbox = [tuple(v + rng.uniform(-a, a) for v in b) for b in t.bbox]
                t.loose = [tuple(v + rng.uniform(-a, a) for v in b) for b in t.loose]
            elif st == "scale":
                s = rng.choice([0.5, 0.97, 1.03, 2.0])
                t.bbox = [tuple(v * s for v in b) for b in t.bbox]
                t.loose = [tuple(v * s for v in b) for b in t.loose]
                t.size = [v * s for v in t.size]
                t.page_w, t.page_h = t.page_w * s, t.page_h * s
            elif st == "angle":
                q = math.radians(rng.choice([90, 180, 270]))
                lo, hi = sorted(rng.sample(range(n + 1), 2)) if n > 1 else (0, n)
                if rng.random() < 0.5:
                    lo, hi = 0, n
                t.angle = [a + q if lo <= i < hi else a for i, a in enumerate(t.angle)]
            elif st == "fonts":
                t.fonts = [rng.choice(FONTS) if rng.random() < 0.5 else f for f in t.fonts]
                if rng.random() < 0.2:
                    t.font = [f + (5 if rng.random() < 0.05 else 0) for f in t.font]      # index out of range
                t.flags = [f | (lv.ITALIC if rng.random() < 0.1 else 0) for f in t.flags]
                t.weight = [rng.choice([w, w, 700, 300]) for w in t.weight]
            elif st == "nonewline":
                t.ch = [" " if c in "\r\n" and rng.random() < 0.3 else c for c in t.ch]
            elif st == "truncate" and n > 2:
                k = rng.randint(1, n - 1)
                for f in ("ch", "bbox", "loose", "size", "weight", "angle", "font", "flags"):
                    setattr(t, f, getattr(t, f)[:k])
            elif st == "degenerate":
                for i in rng.sample(range(n), min(n, max(1, n // 50))):
                    kind = rng.choice(["nan", "huge", "off", "zero", "flat"])
                    if kind == "nan":
                        t.bbox[i] = (float("nan"), 0.0, 1.0, 1.0)
                    elif kind == "huge":
                        t.loose[i] = (0.0, 0.0, 3e6, 1.0)
                    elif kind == "off":
                        t.bbox[i] = t.loose[i] = (-500.0, -500.0, -490.0, -490.0)
                    elif kind == "zero":
                        t.bbox[i] = t.loose[i] = (100.0, 100.0, 100.0, 100.0)
                    else:
                        b = t.bbox[i]
                        t.bbox[i] = (b[0], b[1], b[2], b[1])
                        t.loose[i] = (b[0], b[1], b[2], b[1])
            elif st == "rotation":
                t.rotation = rng.choice([90, 180, 270])
        out.append((pno, roundtrip(t)))
    return out, recipe


def edge_documents(rng):
    docs = []

    def mk(chars, page_no=1, w=595.0, h=842.0, rotation=0, size=10.0):
        pg = Page(rng, w, h, rotation, page_no)
        if chars:
            pg.row(chars, 72, 100, size, "Times-Roman", loose_mode="normal")
        return pg.table()

    empty = mk("")
    docs.append(("empty", [(1, empty, "")]))
    docs.append(("one_char", [(1, mk("Α"), "Α")]))
    docs.append(("one_digit", [(3, mk("7"), "7")]))
    docs.append(("spaces", [(2, mk("    "), "")]))
    nl = Page(rng, 595.0, 842.0, 0, 1)
    nl.newline("\r\n\r\n\n")
    docs.append(("newlines", [(1, nl.table(), "")]))
    docs.append(("plain_no_table", [(1, empty, "Κείμενο χωρίς πίνακα χαρακτήρων\nδεύτερη γραμμή")]))
    t = mk("Πίνακας χωρίς κείμενο")
    docs.append(("table_no_plain", [(1, t, "")]))
    docs.append(("one_huge_char", [(1, mk("Ω", size=400.0), "Ω")]))
    docs.append(("tiny_page", [(1, mk("abc", w=10.0, h=10.0, size=2.0), "abc")]))
    docs.append(("rot_odd", [(1, mk("Σελίδα με περιστροφή", rotation=45), "Σελίδα με περιστροφή")]))
    return docs


def plain_for(rng, t, mode=None):
    text = lv.fixture_text(t)
    mode = mode or rng.choices(["same", "edit", "space", "empty"], [85, 9, 4, 2])[0]
    if mode == "same" or not text:
        return text
    if mode == "empty":
        return ""
    chars = list(text)
    if mode == "edit":
        for _ in range(rng.randint(1, 6)):
            k = rng.randrange(len(chars) + 1)
            op = rng.random()
            if op < 0.4 and k < len(chars):
                del chars[k]
            elif op < 0.8:
                chars.insert(k, rng.choice(GREEK + LATIN + DIGITS))
            elif k < len(chars):
                chars[k] = chars[k].swapcase()
    else:
        chars = [(" " if c == "\n" and rng.random() < 0.3 else c) for c in chars]
        chars = [("\n" if c == " " and rng.random() < 0.05 else c) for c in chars]
    return "".join(chars).strip()


def fuzz_documents(seed, n_synth, real_docs=(), n_perturbed=0, edge=True):
    """real_docs: [(name, [(page, CharTable)])] tables of real documents for the perturbed kind."""
    rng = random.Random(seed)
    if edge:
        for name, pages in edge_documents(rng):
            yield {"name": f"edge/{name}", "kind": "edge", "pages": pages, "context": [(p, t) for p, t, _ in pages]}
    for i in range(n_synth):
        tables = synth_document(rng, i)
        pages = [(p, t, plain_for(rng, t)) for p, t in tables]
        yield {"name": f"synth/{seed}/{i}", "kind": "synth", "pages": pages, "context": tables}
    real_docs = list(real_docs)
    for i in range(n_perturbed if real_docs else 0):
        name, tables = real_docs[rng.randrange(len(real_docs))]
        tables, recipe = perturb_tables(rng, tables)
        chosen = rng.sample(tables, min(len(tables), rng.randint(1, 6)))
        pages = [(p, t, plain_for(rng, t)) for p, t in sorted(chosen, key=lambda x: x[0])]
        yield {"name": f"perturbed/{seed}/{i}/{name}/{recipe}", "kind": "perturbed", "pages": pages,
               "context": tables}
