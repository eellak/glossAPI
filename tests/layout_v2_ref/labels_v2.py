"""Layout v2 (K5g): labels on the plain reading order. Pure Python, standard library only.

The plain arm's text and order are kept exactly. v2 only cuts the plain text into blocks at its own line breaks and
labels each block: title, section_header (with level), paragraph, list_item, caption, footnote, page_header,
page_footer, table, picture, formula. Char geometry comes from the `.k5chars` fixture of the same PDFium text page.

Invariant (asserted per page in `label_page`): the blocks partition the plain text contiguously; blocks are cut only
at plain newlines or at whitespace where PDFium joined two visual rows, so block texts + the original separators ==
plain text, byte for byte, and the scorer's normalisation of "\\n".join(block texts) equals that of the plain text
(asserted in arm_ownlabels_v2). Furniture is labelled, never deleted. No lexicon.

Pipeline per page:
  1. units      PDFium UTF-16 code units -> code points (surrogate pairs joined), each with a box in the page's
                reading frame (the dominant char angle, quantised to 0/90/180/270; /Rotate is not needed because
                char angles are in user space) and an effective size (the loose box extent across the baseline)
  2. align      plain non-space chars <-> unit non-space chars (identity when equal, else difflib opcodes)
  3. lines      one record per plain line: box, size, bold share, math share, angle class, cell gaps, row mates
  4. furniture  top/bottom band lines: page-number pattern on an outermost line, or a signature (digits -> #)
                repeated on >= 30% of the document's sampled pages (document pass over fixtures + context)
  5. tables     runs of cell-like lines (row mates in stream order, or >= 2 cell gaps, or 1 gap and numeric)
  6. blocks     consecutive lines are cut by leading, style, bullets, captions, note marks, short last lines,
                indents and horizontal jumps
  7. labels     caption / list_item / formula / section_header (+ level) / title / footnote area / paragraph
Every rule id (V-nn) is in RULES.md with the tuning page that motivated it.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass, field

# ---- constants -----------------------------------------------------------------------------------------------------
GENERATED, SOFT_HYPHEN, MAP_ERROR, ITALIC, SYMBOLIC, INVISIBLE = 1, 2, 4, 8, 16, 32
CELL_GAP = 1.5            # a horizontal gap > CELL_GAP x size inside one plain line is a cell gap
LEADING = 1.5             # paragraph break when the baseline step > LEADING x the page leading (size-scaled)
SIZE_STEP = 0.10          # style break when line sizes differ by > 10%
HEADING = 1.15            # heading size ratio to the body size
FURN_BAND, FURN_SHARE, FURN_MIN, FURN_TOL = 0.13, 0.30, 3, 6.0
FOOT_SIZE = 0.9
BOLD_NAME = re.compile(r"bold|black|heavy|semibold|demi|-b$|,b$|-bd|cmbx|^grx|bx\d", re.I)
ITALIC_NAME = re.compile(r"italic|oblique|-it$|,it$", re.I)
MATH_FONT = re.compile(r"cmmi|cmsy|cmex|msam|msbm|symbol|math|mt ?extra|euclid|rsfs|wasy|stmary", re.I)
BULLET = re.compile(r"^\s*[•–▪■□◦○●➢►▶✓✔·\-\*-](\s|$)")
ENUM = re.compile(r"^\s*(\(?([0-9]{1,3}|[a-zα-ω]|[ivx]{1,4})[.)](\s|$)|\(?[a-zα-ω]{1,2}[΄']?\)(\s|$)"
                  r"|[A-ZΑ-Ω][.)]\s+(?![A-ZΑ-Ω]\.)\S|[0-9]{1,2}\s?[a-zα-ω]\s?[.)]\s|o\s+[A-ZΑ-Ω])")
NUMBERED = re.compile(r"^\s*(\d+(\.\d+)*\.?\s|[IVX]{1,4}\.\s|ΚΕΦΑΛΑΙΟ|Κεφάλαιο|Chapter|CHAPTER|"
                      r"ΠΑΡΑΡΤΗΜΑ|Παράρτημα|Appendix|ΜΕΡΟΣ|Μέρος)")
SECNUM = re.compile(r"^\s*([0-9]{1,2}|[IVXΙ]{1,4}|[ΑΒΓΔΕΖΗΘ])(\.[0-9]{1,2})+\.?\s+\S")
CAPTION = re.compile(r"^\s*(Σχήμα|ΣΧΗΜΑ|Σχ\.|Εικόνα|ΕΙΚΟΝΑ|Εικ\.|Πίνακας|ΠΙΝΑΚΑΣ|Πιν\.|Πίν\.|Διάγραμμα|ΔΙΑΓΡΑΜΜΑ|"
                     r"Γράφημα|ΓΡΑΦΗΜΑ|Χάρτης|ΧΑΡΤΗΣ|Φωτογραφία|Figure|FIGURE|Fig\.|Table|TABLE|Chart|Graph|Map|Photo)"
                     r"\s*[0-9Α-ΩA-Z]")
FOOTMARK = re.compile(r"^\s*([0-9¹²³⁴⁵⁶⁷⁸⁹⁰]{1,3}|\*|†|‡)")
PAGE_NUM = re.compile(r"^[-–—·\[(]?\s*(\d{1,4}|[ivxlcdm]{1,7}|[α-ω]{1,3}[΄ʹ'])\s*[-–—·\])]?$", re.I)
OPS = set("=+−×÷±∑√≤≥≈≠∂∈∉⊂⊆∪∩⊗⊕→⇒⇔∀∃∇∫")
LABELS = ("title", "section_header", "paragraph", "list_item", "caption", "footnote", "page_header", "page_footer",
          "table", "picture", "formula")


def median_low(xs):
    xs = sorted(xs)
    return xs[(len(xs) - 1) // 2] if xs else 0.0


def q(v):                      # inputs rounded to 1/64 pt, half away from zero (Rust f64::round semantics)
    return math.floor(v * 64 + 0.5) / 64 if v >= 0 else -math.floor(-v * 64 + 0.5) / 64


# ---- 1. units ------------------------------------------------------------------------------------------------------
@dataclass
class Unit:
    ch: str
    geo: tuple | None        # (x0, y0, x1, y1) reading frame, top-down; None if degenerate
    size: float
    bold: bool
    math: bool
    angle: int               # quantised char angle in degrees (0/90/180/270)
    italic: bool = False
    weight: int = 0


def utf16_units(chs):
    i, n = 0, len(chs)
    while i < n:
        c = chs[i]
        if "\ud800" <= c <= "\udbff" and i + 1 < n and "\udc00" <= chs[i + 1] <= "\udfff":
            yield i, chr(0x10000 + ((ord(c) - 0xD800) << 10) + (ord(chs[i + 1]) - 0xDC00))
            i += 2
            continue
        yield i, ("�" if "\ud800" <= c <= "\udfff" else c)
        i += 1


def quant_angle(a):
    """Radians -> 0/90/180/270 (nearest quarter turn; ties go to the lower multiple)."""
    d = math.degrees(a) % 360.0
    k = int(math.floor(d / 90.0 + 0.5)) % 4
    return 90 * k


def _frame(angle):
    """Unit vectors (d, u) of the reading frame of a char angle: d = baseline direction, u = glyph up, in PDF user
    space (y up). PDFium's char angle runs clockwise (text running upwards reports 270), so the frame turns by
    -angle."""
    return {0: ((1, 0), (0, 1)), 270: ((0, 1), (-1, 0)), 180: ((-1, 0), (0, -1)), 90: ((0, -1), (1, 0))}[angle]


def _to_frame(box, angle, origin):
    """PDF user-space box (l, b, r, t) -> reading-frame box (x0, top, x1, bottom), y growing downwards."""
    (dx, dy), (ux, uy) = _frame(angle)
    l, b, r, t = box
    xs = [px * dx + py * dy for px, py in ((l, b), (r, t))]
    ys = [px * ux + py * uy for px, py in ((l, b), (r, t))]
    ox, oy = origin
    return min(xs) - ox, oy - max(ys), max(xs) - ox, oy - min(ys)


def frames_of(t):
    """{angle: (origin, W, H)}: each quarter-turn frame's origin and page extent."""
    out = {}
    corners = [(0.0, 0.0), (t.page_w, 0.0), (0.0, t.page_h), (t.page_w, t.page_h)]
    for a in (0, 90, 180, 270):
        (dx, dy), (ux, uy) = _frame(a)
        fx = [px * dx + py * dy for px, py in corners]
        fy = [px * ux + py * uy for px, py in corners]
        out[a] = ((min(fx), max(fy)), max(fx) - min(fx), max(fy) - min(fy))
    return out


def page_units(t):
    """-> (units, frames, display angle, majority angle). Each unit's box is in the frame of its own char angle, so
    lines of rotated content (landscape tables, margin notes) get upright geometry; /Rotate only decides which
    angle is upright on the display (`disp`), which is used for page furniture."""
    raw = list(utf16_units(t.ch))
    frames = frames_of(t)
    disp = (360 - (t.rotation or 0)) % 360
    counts = {0: 0, 90: 0, 180: 0, 270: 0}
    units = []
    for i, c in raw:
        a = quant_angle(t.angle[i])
        name = t.fonts[t.font[i]] if t.font[i] < len(t.fonts) else ""
        bold = bool(BOLD_NAME.search(name.split("+")[-1]))
        mth = bool(MATH_FONT.search(name))
        ital = bool(t.flags[i] & ITALIC) or bool(ITALIC_NAME.search(name.split("+")[-1]))
        wt = int(t.weight[i])
        bb, lo = t.bbox[i], t.loose[i]
        if c.isspace() or any(v != v or abs(v) > 1e6 for v in bb + lo):
            units.append(Unit(c, None, 0.0, bold, mth, a, ital, wt))
            continue
        origin, W, H = frames[a]
        bb = tuple(q(v) for v in bb)
        lo = tuple(q(v) for v in lo)
        x0, y0, x1, y1 = _to_frame(bb, a, origin)
        lx0, ly0, lx1, ly1 = _to_frame(lo, a, origin)
        if not (0 <= (x0 + x1) / 2 <= W and 0 <= (y0 + y1) / 2 <= H):
            units.append(Unit(c, None, 0.0, bold, mth, a, ital, wt))   # outside the page box: not printed
            continue
        size = (ly1 - ly0) if ly1 > ly0 else ((y1 - y0) if y1 > y0 else abs(t.size[i]))
        if lx1 > lx0:
            x0, x1 = lx0, lx1
        if ly1 > ly0:
            y0, y1 = ly0, ly1
        counts[a] += 1
        units.append(Unit(c, (x0, y0, x1, y1), size, bold, mth, a, ital, wt))
    major = max(counts, key=lambda k: (counts[k], k == disp, -k))
    return units, frames, disp, major


# ---- 2. align ------------------------------------------------------------------------------------------------------
def align(plain, units):
    """plain index -> unit index (or -1) for non-space plain chars; equal non-space sequences map 1:1 (the normal
    case: plain is get_text_range of the same text page); otherwise difflib equal opcodes."""
    pi = [k for k, c in enumerate(plain) if not c.isspace()]
    ui = [k for k, u in enumerate(units) if not u.ch.isspace()]
    ps = "".join(plain[k] for k in pi)
    us = "".join(units[k].ch for k in ui)
    m = [-1] * len(plain)
    if ps == us:
        for a, b in zip(pi, ui):
            m[a] = b
        return m, 0
    sm = difflib.SequenceMatcher(None, ps, us, autojunk=False)
    matched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for o in range(i2 - i1):
                m[pi[i1 + o]] = ui[j1 + o]
            matched += i2 - i1
    return m, len(ps) - matched


# ---- 3. lines ------------------------------------------------------------------------------------------------------
@dataclass
class PLine:
    k: int
    start: int
    end: int
    text: str
    n: int = 0
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    size: float = 0.0
    bold: float = 0.0
    math: float = 0.0
    angle: int = 0
    gaps: list = field(default_factory=list)
    emph: float = 0.0            # share of chars set off from the page's running text: bold name, italic, heavier
    wts: dict = field(default_factory=dict)   # font weight -> char count (PDFium GetFontWeight)
    rows: int = 1                # visual rows inside this plain line (PDFium may join rows without a newline)
    fx0: float = 0.0             # first row: left edge, bottom; last row: right edge, top, bottom
    fy1: float = 0.0
    lx1: float = 0.0
    ly0: float = 0.0
    ly1: float = 0.0
    mate_prev: bool = False      # shares a visual row with the previous plain line (to its right)
    furniture: str | None = None
    furn_reason: str | None = None

    @property
    def geo(self):
        return self.n > 0

    @property
    def yc(self):
        return (self.y0 + self.y1) / 2


def split_lines(plain):
    out, pos = [], 0
    for k, s in enumerate(plain.split("\n")):
        out.append(PLine(k, pos, pos + len(s), s))
        pos += len(s) + 1
    return out


LINE_START = re.compile(r"^\s*([-•·▪■□◦○●➢►▶\u2022\u25aa\uf000-\uf0ff*#|>_`](\s|$)|\|?\s*:?-{3,}|[#|*`>_])")


def split_rows(lines, plain, units, amap):
    """PDFium sometimes joins visual rows without a newline. Split a plain line where the stream starts a new row
    (a full-size char whose centre is > 0.5 x size off the current row) and plain has whitespace there; the cut
    falls on that whitespace, so no text moves and the scorer's whitespace collapse sees the same text. Never cut
    where the new piece would start with something the scorer treats specially at a line start (bullets, table
    rules, Markdown marks), nor inside a line that starts with a table rule."""
    out = []
    for ln in lines:
        pos = [i for i in range(ln.start, ln.end) if amap[i] >= 0 and units[amap[i]].geo]
        cuts = []
        if len(pos) >= 2 and not re.match(r"^\s*\|?\s*:?-{3,}", ln.text):
            sz = median_low([units[amap[i]].size for i in pos])
            cur, prev = None, None
            for i in pos:
                u = units[amap[i]]
                if u.size < 0.8 * sz:
                    prev = i
                    continue
                yc = (u.geo[1] + u.geo[3]) / 2
                if cur is not None and (abs(yc - cur[0]) > 0.5 * sz or u.angle != cur[1]) and prev is not None:
                    gap = plain[prev + 1:i]
                    if gap and gap.isspace() and not LINE_START.match(plain[i:ln.end]):
                        cuts.append((prev + 1, i))
                if cur is None or abs(yc - cur[0]) > 0.5 * sz or u.angle != cur[1]:
                    cur = (yc, u.angle)
                prev = i
        a = ln.start
        for e, b in cuts:
            out.append(PLine(len(out), a, e, plain[a:e]))
            a = b
        out.append(PLine(len(out), a, ln.end, plain[a:ln.end]))
    return out


def line_geometry(lines, plain, units, amap, disp):
    for ln in lines:
        us = [units[amap[i]] for i in range(ln.start, ln.end) if amap[i] >= 0 and units[amap[i]].geo]
        if not us:
            continue
        acount = {}
        for u in us:
            acount[u.angle] = acount.get(u.angle, 0) + 1
        ln.angle = max(acount, key=lambda a: (acount[a], a == disp, -a))
        main = [u for u in us if u.angle == ln.angle]
        ln.n = len(main)
        ln.x0 = min(u.geo[0] for u in main)
        ln.x1 = max(u.geo[2] for u in main)
        sz = median_low([u.size for u in main])
        big = [u for u in main if u.size >= 0.8 * sz] or main        # sub/superscripts do not set the box
        rows = []                     # visual rows in stream order: a new row when the char drops below the row
        for u in big:
            yc = (u.geo[1] + u.geo[3]) / 2
            if rows and abs(yc - rows[-1][0]) <= 0.5 * sz:
                rows[-1][1].append(u)
            elif rows and any(abs(yc - r[0]) <= 0.5 * sz for r in rows):
                next(r for r in rows if abs(yc - r[0]) <= 0.5 * sz)[1].append(u)
            else:
                rows.append([yc, [u]])
        rows.sort(key=lambda r: r[0])
        r0, r1 = rows[0][1], rows[-1][1]
        ln.rows = len(rows)
        ln.y0 = median_low([u.geo[1] for u in r0])
        ln.fy1 = median_low([u.geo[3] for u in r0])
        ln.ly0 = median_low([u.geo[1] for u in r1])
        ln.y1 = ln.ly1 = median_low([u.geo[3] for u in r1])
        ln.fx0 = min(u.geo[0] for u in r0)
        ln.lx1 = max(u.geo[2] for u in r1)
        ln.size = sz
        ln.bold = sum(u.bold for u in main) / len(main)
        ln.emph = sum(u.bold or u.italic for u in main) / len(main)
        for u in main:
            ln.wts[u.weight] = ln.wts.get(u.weight, 0) + 1
        ln.math = sum(u.math for u in main) / len(main)
        if ln.rows == 1:
            seq = sorted(main, key=lambda u: u.geo[0])
            for a, b in zip(seq, seq[1:]):
                if b.geo[0] - a.geo[2] > CELL_GAP * max(sz, 1.0):
                    ln.gaps.append((a.geo[2], b.geo[0]))
    prev = None
    for ln in lines:
        if not ln.geo:
            continue
        if prev is not None and prev.angle == ln.angle:
            h = min(prev.ly1 - prev.ly0, ln.fy1 - ln.y0)
            ov = min(prev.ly1, ln.fy1) - max(prev.ly0, ln.y0)
            if h > 0 and ov >= 0.5 * h and ln.fx0 >= prev.lx1 - 0.5 * ln.size:
                ln.mate_prev = True
        prev = ln
    return lines


# ---- document pass -------------------------------------------------------------------------------------------------
def norm_sig(text):
    t = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", re.sub(r"\d", "#", t)).strip()


@dataclass
class PageGeo:
    plain: str
    lines: list
    frames: dict          # angle -> (origin, W, H)
    disp: int             # char angle that is upright on the display (from /Rotate)
    major: int            # char angle of most characters
    unmatched: int
    units: int

    def W(self, ln=None):
        return self.frames[ln.angle if ln is not None else self.disp][1]

    def H(self, ln=None):
        return self.frames[ln.angle if ln is not None else self.disp][2]


def page_geo(plain, t):
    units, frames, disp, major = page_units(t)
    amap, unmatched = align(plain, units)
    lines = line_geometry(split_rows(split_lines(plain), plain, units, amap), plain, units, amap, disp)
    return PageGeo(plain, lines, frames, disp, major, unmatched, len(units))


def fixture_text(t):
    """The plain arm's text rebuilt from the char table (get_text_range + the arm's newline normalisation)."""
    s = "".join(c for _, c in utf16_units(t.ch))
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def band_of(ln, g):
    if ln.angle != g.disp:
        return None
    H = g.H(ln)
    if ln.y1 <= FURN_BAND * H:
        return "top"
    if ln.y0 >= (1 - FURN_BAND) * H:
        return "bottom"
    return None


@dataclass
class DocContext:
    body: float
    heading_sizes: list
    furniture: list           # {band, signature, y, pages}
    pages: int


def size_key(s):
    return round(s * 2) / 2


def document_context(geos):
    """geos: {page_no: PageGeo} of the document's sampled pages (fixtures + context)."""
    sizes = {}
    for g in geos.values():
        for ln in g.lines:
            if ln.geo:
                k = size_key(ln.size)
                sizes[k] = sizes.get(k, 0) + ln.n
    body = max(sizes, key=lambda k: (sizes[k], -k)) if sizes else 10.0
    hs = {}
    for g in geos.values():
        for ln in g.lines:
            if (ln.geo and HEADING * body <= ln.size <= 4 * body and ln.x1 - ln.x0 <= 0.7 * g.W(ln)
                    and sum(c.isalpha() for c in ln.text) >= 3):
                k = size_key(ln.size)
                hs[k] = hs.get(k, 0) + 1
    heading_sizes = sorted(hs, reverse=True)[:3]
    texted = [pg for pg, g in geos.items() if any(ln.geo for ln in g.lines)]
    need = max(FURN_MIN, math.ceil(FURN_SHARE * len(texted)))
    need_par = [max(FURN_MIN, math.ceil(FURN_SHARE * sum(1 for pg in texted if pg % 2 == k))) for k in (0, 1)]
    occ = {}
    for pg, g in geos.items():
        for ln in g.lines:
            if ln.geo and ln.text.strip():
                band = band_of(ln, g)
                if band:
                    occ.setdefault((band, norm_sig(ln.text)), []).append((pg, ln.yc))
    furniture = []
    for (band, sig), lst in sorted(occ.items()):
        seen = []
        for pg, yc in lst:
            pages = {p for p, y in lst if abs(y - yc) <= FURN_TOL}
            same = {p for p in pages if p % 2 == pg % 2}
            if (len(pages) >= need or len(same) >= need_par[pg % 2]) and not any(abs(yc - y) <= 0.01 for y in seen):
                seen.append(yc)
                furniture.append({"band": band, "signature": sig, "y": round(yc, 2), "pages": len(pages)})
    return DocContext(body, heading_sizes, furniture, len(texted))


# ---- per page: furniture, tables, blocks, labels -------------------------------------------------------------------
@dataclass
class Block:
    first: int                # first plain line index
    last: int                 # last plain line index (inclusive)
    label: str = "paragraph"
    level: int | None = None
    text: str = ""
    bbox: list | None = None  # [x0, top, x1, bottom] in the block's reading frame
    why: str = ""
    rotated: int = 0          # clockwise quarter turns of the block's text against the display (degrees)
    cut: str | None = None    # why this block was cut from the previous one (cut_reason)


@dataclass
class PageLabels:
    blocks: list
    warnings: list
    rotation: int


def mark_furniture(g, ctx):
    geo = [ln for ln in g.lines if ln.geo and ln.text.strip() and ln.angle == g.disp]
    if not geo:
        return
    top = min(geo, key=lambda l: (l.y0, l.k))
    bot = max(geo, key=lambda l: (l.y1, -l.k))
    for ln in geo:
        band = band_of(ln, g)
        if not band:
            continue
        txt = ln.text.strip()
        if (ln is top or ln is bot) and PAGE_NUM.match(txt):
            ln.furniture, ln.furn_reason = band, "page_number"
            continue
        sig = norm_sig(txt)
        for f in ctx.furniture:
            if f["band"] == band and f["signature"] == sig and abs(f["y"] - ln.yc) <= FURN_TOL:
                ln.furniture, ln.furn_reason = band, "repeated"
                break
    for ln in geo:                 # a repeated line set tight against non-furniture text continues that text
        if ln.furn_reason != "repeated":
            continue
        for o in geo:
            if o is ln or o.furniture or min(o.x1, ln.x1) <= max(o.x0, ln.x0):
                continue
            gap = max(o.y0 - ln.y1, ln.y0 - o.y1)
            if gap < 0.5 * min(o.size, ln.size):
                ln.furniture, ln.furn_reason = None, None
                break


def numeric_share(text):
    t = [c for c in text if not c.isspace()]
    return sum(c.isdigit() for c in t) / max(1, len(t))


def table_runs(g):
    """Plain-line index ranges [a, b] labelled table. Strong cell lines: >= 2 cell gaps, or 1 gap and numeric, or a
    row mate separated by a cell gap. Weak cell lines: 1 cell gap, or a left edge at a cell start (gap end) of a
    strong line within 3 lines. A run = strong/weak lines with at most one other line between two of them; it needs
    >= 2 strong lines, >= 3 cell lines and >= 2 visual rows; weak lines at its ends are trimmed."""
    L = g.lines
    n = len(L)
    ok = [ln.geo and not ln.furniture for ln in L]
    strong = [False] * n
    for k, ln in enumerate(L):
        if not ok[k]:
            continue
        mate_gap = ln.mate_prev and L[k - 1].geo and ln.fx0 - L[k - 1].lx1 > CELL_GAP * ln.size
        nxt = L[k + 1] if k + 1 < n and ok[k + 1] else None
        next_gap = nxt is not None and nxt.mate_prev and nxt.fx0 - ln.lx1 > CELL_GAP * ln.size
        strong[k] = ln.math < 0.5 and (len(ln.gaps) >= 2 or (len(ln.gaps) >= 1 and numeric_share(ln.text) >= 0.3)
                                       or bool(mate_gap) or next_gap)
    def overlaps(a, b):
        return any(min(g[1], h[1]) > max(g[0], h[0]) for g in a.gaps for h in b.gaps)
    for k, ln in enumerate(L):            # one-gap lines whose gaps line up with a neighbour's: a two-column table
        if ok[k] and not strong[k] and len(ln.gaps) == 1 and ln.math < 0.3:
            nb = [L[j] for j in (k - 2, k - 1, k + 1, k + 2) if 0 <= j < n and ok[j] and L[j].gaps and L[j].math < 0.3
                  and L[j].angle == ln.angle]
            if sum(overlaps(ln, o) for o in nb) >= 2:
                strong[k] = True
    weak = [False] * n
    for k, ln in enumerate(L):
        if not ok[k] or strong[k]:
            continue
        if len(ln.gaps) == 1:
            weak[k] = True
            continue
        for j in range(max(0, k - 3), min(n, k + 4)):
            if strong[j] and L[j].angle == ln.angle and any(abs(ln.fx0 - gx1) <= 0.5 * ln.size for _, gx1 in L[j].gaps):
                weak[k] = True
                break
    cell = [a or b for a, b in zip(strong, weak)]
    runs, k = [], 0
    while k < n:
        if not strong[k]:
            k += 1
            continue
        j, ang = k, L[k].angle

        def c(i):
            return cell[i] and L[i].angle == ang
        def filler(i):                     # one other line between cell lines: a short line (a wrapped cell)
            return ok[i] and L[i].angle == ang and L[i].x1 - L[i].x0 <= 0.5 * g.W(L[i])
        while j + 1 < n and (c(j + 1) or (j + 2 < n and c(j + 2) and filler(j + 1))):
            j += 1 if c(j + 1) else 2
        while j > k and not strong[j]:
            j -= 1
        ls = [L[i] for i in range(k, j + 1) if L[i].geo]
        rows = 1 + sum(1 for ln in ls[1:] if not ln.mate_prev)
        wide = sum(len(L[i].gaps) >= 3 for i in range(k, j + 1))
        if sum(strong[k:j + 1]) >= 2 and (sum(cell[k:j + 1]) >= 3 or wide >= 2) and rows >= 2:
            runs.append((k, j))
        k = j + 1
    runs = sorted(runs + column_runs(g, ok, runs))
    out = []                     # grow each run over adjacent cell lines of its angle, then merge touching runs
    for a, b in runs:
        ang = L[a].angle
        while a - 1 >= 0 and cell[a - 1] and L[a - 1].angle == ang:
            a -= 1
        while b + 1 < n and cell[b + 1] and L[b + 1].angle == ang:
            b += 1
        if out and a <= out[-1][1] + 1 and L[out[-1][0]].angle == ang:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def column_runs(g, ok, taken):
    """Column-ordered tables (one cell per plain line, the stream reading down each column): a run of >= 6 narrow
    lines (<= 25% of the frame width, <= 40 chars) of one angle with >= 2 upward jumps (a new column) and >= 4
    downward steps."""
    L = g.lines
    used = {i for a, b in taken for i in range(a, b + 1)}
    def narrow(i):
        ln = L[i]
        return ok[i] and i not in used and ln.x1 - ln.x0 <= 0.25 * g.W(ln) and len(ln.text.strip()) <= 40
    out, k, n = [], 0, len(L)
    while k < n:
        if not narrow(k):
            k += 1
            continue
        j = k
        while j + 1 < n and narrow(j + 1) and L[j + 1].angle == L[k].angle:
            j += 1
        up = down = 0
        for i in range(k + 1, j + 1):
            a, b = L[i - 1], L[i]
            if b.y0 < a.y0 - 1.5 * b.size:
                up += 1
            elif b.y0 > a.y1 - 0.5 * b.size:
                down += 1
        if j - k + 1 >= 6 and up >= 2 and down >= 4:
            out.append((k, j))
        k = j + 1
    return out


def starts_list(text):
    return bool(BULLET.match(text) or ENUM.match(text))


LOOKALIKE = str.maketrans({"∆": "Δ", "µ": "μ", "Ω": "Ω"})


def is_caption(text):
    return bool(CAPTION.match(text.translate(LOOKALIKE)))


FIGWORD = re.compile(r"^\s*(Σχήμα|ΣΧΗΜΑ|Σχ\.|Εικόνα|ΕΙΚΟΝΑ|Εικ\.|Διάγραμμα|ΔΙΑΓΡΑΜΜΑ|Γράφημα|ΓΡΑΦΗΜΑ|Χάρτης|ΧΑΡΤΗΣ|"
                     r"Φωτογραφία|Figure|FIGURE|Fig\.|Chart|Graph|Map|Photo)")


def is_formula(lines):
    t = [c for ln in lines for c in ln.text if not c.isspace()]
    if len(t) < 4 or len(set(t)) < 3 or sum(c.isdigit() or c in ",." for c in t) >= 0.6 * len(t):
        return False
    mathc = sum(ln.math * ln.n for ln in lines)
    sym = sum(not c.isalnum() for c in t)
    ops = any(c in OPS for c in t)
    return mathc >= 0.5 * max(1, sum(ln.n for ln in lines)) or (ops and sym >= 0.4 * len(t) and not t[0].isalpha())


def label_page(plain, t, ctx, page_no, geo=None):
    """Blocks of one page over the plain text (the order is the plain order; nothing is moved or dropped)."""
    g = geo or page_geo(plain, t)
    warnings = []
    if g.unmatched:
        warnings.append({"kind": "unaligned_chars", "count": g.unmatched})
    rel = (g.major - g.disp) % 360
    if rel:
        warnings.append({"kind": "rotated_content", "degrees_cw": rel})
    rot_lines = sum(1 for ln in g.lines if ln.geo and ln.angle != g.disp)
    if rot_lines:
        warnings.append({"kind": "rotated_lines", "count": rot_lines})
    if t.rotation:
        warnings.append({"kind": "page_rotation", "degrees": t.rotation})
    L = g.lines
    if not plain:
        return PageLabels([], warnings, rel)
    mark_furniture(g, ctx)
    body = ctx.body
    sizes = {}
    for ln in L:
        if ln.geo and not ln.furniture:
            k = size_key(ln.size)
            sizes[k] = sizes.get(k, 0) + ln.n
    if sizes:
        own = max(sizes, key=lambda k: (sizes[k], -k))
        at_body = sum(c for k, c in sizes.items() if abs(k - body) <= 0.15 * body)
        if abs(own - body) > 0.15 * body and sizes[own] >= 0.4 * sum(sizes.values()) and \
                at_body < 0.15 * sum(sizes.values()):
            body = own
    steps = []
    gl = [ln for ln in L if ln.geo and not ln.furniture]
    for a, b in zip(gl, gl[1:]):
        if a.angle != b.angle:
            continue
        d = b.fy1 - a.ly1
        if 0.5 * a.size < d < 3 * a.size and abs(a.size - body) <= 0.15 * body:
            steps.append(d)
    leading = median_low(steps) if steps else 1.2 * body
    wc = {}
    for ln in L:
        if ln.geo and not ln.furniture:
            for w, c in ln.wts.items():
                if w > 0:
                    wc[w] = wc.get(w, 0) + c
    wmode = max(wc, key=lambda w: (wc[w], -w)) if wc else 0
    if wmode:                                  # heavier than the page's running text by >= 40 counts as emphasis
        for ln in L:
            if ln.geo:
                heavy = sum(c for w, c in ln.wts.items() if w >= wmode + 40) / ln.n
                ln.emph = max(ln.emph, heavy)
    runs = table_runs(g)
    in_table = {}
    for a, b in runs:
        for i in range(a, b + 1):
            in_table[i] = (a, b)

    # -- cut blocks ------------------------------------------------------------------------------------------------
    blocks = []
    cur = None

    def close():
        nonlocal cur
        if cur is not None:
            blocks.append(cur)
        cur = None

    def lines_of(b):
        return [L[i] for i in range(b.first, b.last + 1)]

    def geo_lines(b):
        return [ln for ln in lines_of(b) if ln.geo]

    k = 0
    while k < len(L):
        ln = L[k]
        if k in in_table:
            close()
            a, b = in_table[k]
            tl = [L[i] for i in range(a, b + 1) if L[i].geo]
            mshare = sum(l.math * l.n for l in tl) / max(1, sum(l.n for l in tl))
            if mshare >= 0.3:
                blocks.append(Block(a, b, "formula", why="math_run"))
            else:
                blocks.append(Block(a, b, "table", why="table_run"))
            k = b + 1
            continue
        if ln.furniture:
            close()
            lab = "page_header" if ln.furniture == "top" else "page_footer"
            prev = blocks[-1] if blocks else None
            if (prev is not None and prev.label == lab and prev.last == k - 1 and ln.mate_prev):
                prev.last = k
            else:
                blocks.append(Block(k, k, lab, why=ln.furn_reason))
            k += 1
            continue
        if cur is None:
            cur = Block(k, k)
            k += 1
            continue
        gls = geo_lines(cur)
        a = gls[-1] if gls else None
        if not ln.geo or a is None:
            cur.last = k
            k += 1
            continue
        cut = cut_reason(cur, gls, a, ln, body, leading, g)
        if cut:
            close()
            cur = Block(k, k, cut=cut)
        else:
            cur.last = k
        k += 1
    close()

    # -- label -----------------------------------------------------------------------------------------------------
    glines = [geo_lines(b) for b in blocks]
    for i, b in enumerate(blocks):
        b.text = plain[L[b.first].start:L[b.last].end]
        gls = glines[i]
        if gls:
            b.bbox = [round(min(l.x0 for l in gls), 2), round(min(l.y0 for l in gls), 2),
                      round(max(l.x1 for l in gls), 2), round(max(l.y1 for l in gls), 2)]
            b.rotated = (gls[0].angle - g.disp) % 360
        if b.label != "paragraph" or not gls:
            continue
        iso = []                                  # extra space above / below, in units of the page leading
        for j in (i - 1, i + 1):
            o = glines[j] if 0 <= j < len(blocks) and blocks[j].label not in ("page_header", "page_footer") else []
            if not o or o[0].angle != gls[0].angle:
                iso.append(None)
                continue
            d = (gls[0].fy1 - o[-1].ly1) if j < i else (o[0].fy1 - gls[-1].ly1)
            iso.append(d / leading if d > 0 else None)
        label_block(b, gls, body, ctx, g, iso)
    list_neighbours(blocks, glines)
    picture_runs(blocks, glines, body)
    footnote_area(blocks, L, body, g)
    title_rule(blocks, L, body, ctx, page_no, g)
    # -- the invariant ------------------------------------------------------------------------------------------
    assert blocks[0].first == 0 and blocks[-1].last == len(L) - 1
    assert all(blocks[i].last + 1 == blocks[i + 1].first for i in range(len(blocks) - 1))
    rebuilt = "".join(b.text + plain[L[b.last].end:L[blocks[i + 1].first].start] if i + 1 < len(blocks) else b.text
                      for i, b in enumerate(blocks))
    assert rebuilt == plain, "v2 must not move, add or drop text"
    assert all(plain[L[b.last].end:L[blocks[i + 1].first].start].isspace() for i, b in enumerate(blocks[:-1]))
    return PageLabels(blocks, warnings, rel)


def cut_reason(cur, gls, a, ln, body, leading, g):
    """Why line `ln` starts a new block after the current block (last geometric line `a`), or None."""
    s = max(ln.size, a.size, 1.0)
    first = gls[0]
    if ln.angle != a.angle:
        return "angle"
    if ln.mate_prev and a.k == ln.k - 1:
        return None                                            # same visual row, to the right: PDFium split the row
    if ln.y0 < a.y1 - 0.5 * min(ln.size, a.size):
        return "not_below"
    lead = leading * (ln.size / body if body else 1.0)
    if ln.fy1 - a.ly1 > LEADING * max(lead, 0.9 * ln.size):
        return "leading"
    if abs(ln.size - a.size) > SIZE_STEP * s:
        return "size"
    in_caption = is_caption(first.text)
    if (a.bold >= 0.6) != (ln.bold >= 0.6) and not in_caption:
        return "bold"
    right = max(max(l.x1 for l in gls), ln.x1)
    if BULLET.match(ln.text):
        return "list"
    if ENUM.match(ln.text) and a.lx1 < right - 1.5 * s:     # an enumeration after a line that is not full
        return "list"
    if is_caption(ln.text) and a.lx1 < right - 1.5 * s:
        return "caption"
    if ln.x0 > a.x1 or ln.x1 < a.x0:
        return "no_overlap"
    head_like = ln.size >= HEADING * body or (ln.bold >= 0.6 and a.bold >= 0.6)
    if in_caption:
        return None
    if not head_like and a.lx1 < right - 3 * s:
        return "short_line"
    if head_like:
        if SECNUM.match(ln.text) or (NUMBERED.match(ln.text) and not NUMBERED.match(first.text)):
            return "heading_number"                            # a new numbered heading
        edge = text_edge(g, a)
        if edge is not None and a.lx1 < edge - 6 * s:
            return "heading_short"                             # a heading line well short of the text edge
        return None
    is_list = starts_list(cur_text_first(gls))
    if len(gls) == 1:
        full = first.lx1 >= right - 1.5 * s                    # the first line runs to the right edge
        if ln.fx0 > first.fx0 + 0.8 * s:                       # next line indented under the first
            if (full or is_list) and ln.fx0 - first.fx0 <= 6 * s:
                return None                                    # hanging indent (references, list items)
            return "indent"
        if ln.fx0 < first.fx0 - 0.8 * s:                       # the first line was indented
            if full and first.fx0 - ln.fx0 <= 6 * s:
                return None                                    # first-line indent paragraph
            return "outdent"
        return None
    left = min(l.fx0 for l in gls[1:])
    if ln.fx0 > left + 0.8 * s and not is_list:
        return "indent"
    if ln.fx0 < left - 0.8 * s:
        return "outdent"
    return None


def text_edge(g, a):
    """Right edge of the running text that shares `a`'s left margin: the 90th-percentile right end of the page's
    lines (>= 20 chars, same angle) whose left edge is within 1 size of a's; None if there are fewer than 3."""
    if g.__dict__.get("_edges") is None:
        g._edges = {}
    key = (a.angle, round(a.x0))
    if key not in g._edges:
        ends = sorted(l.lx1 for l in g.lines if l.geo and l.angle == a.angle and len(l.text.strip()) >= 20
                      and abs(l.x0 - a.x0) <= a.size)
        g._edges[key] = ends[int(0.9 * (len(ends) - 1))] if len(ends) >= 3 else None
    return g._edges[key]


def cur_text_first(gls):
    return gls[0].text if gls else ""


def isolated(iso, k=1.4):
    """Extra space on at least one side (> k x leading) and on no side less than a normal line step."""
    above, below = iso
    sides = [x for x in (above, below) if x is not None]
    return bool(sides) and max(sides) > k and min(sides) > 0.8


def label_block(b, gls, body, ctx, g, iso=(None, None)):
    text = b.text.strip()
    if not gls or not text:
        return
    size = median_low([l.size for l in gls])
    letters = sum(c.isalpha() for c in text)
    width = max(l.x1 for l in gls) - min(l.x0 for l in gls)
    bold = sum(l.bold * l.n for l in gls) / max(1, sum(l.n for l in gls))
    if is_caption(text):
        b.label, b.why = "caption", "caption_pattern"
        return
    if is_formula(gls) and letters < 0.5 * len(text.replace(" ", "")):
        b.label, b.why = "formula", "formula"
        return
    nrows = sum(l.rows for l in gls)
    big = size >= HEADING * body
    strong_style = big or bold >= 0.9
    short = (nrows <= (4 if strong_style else 3) and len(text) <= 160 and letters >= 2
             and width <= 0.85 * g.W(gls[0]))
    ends = text[-1:] in ",;" or (text[-1:] == "." and not text.endswith("..") and not (bold >= 0.9 and nrows <= 4))
    sec = SECNUM.match(text)
    numbered = (sec or NUMBERED.match(text)) and nrows == 1 and len(text) <= 120 and not ends and text[-1] != ":"
    if short and not ends and (big or (bold >= 0.6 and abs(size - body) <= 0.15 * body)
                               or (NUMBERED.match(text) and bold >= 0.6)):
        b.label, b.why = "section_header", "big" if big else "bold"
    elif numbered and size >= 0.95 * body and isolated(iso) and (sec or len(text) <= 80):
        b.label, b.why = "section_header", "numbered_isolated"
    elif (nrows == 1 and len(text) <= 80 and letters >= 3 and not ends and text[-1] != ":"
          and min(l.emph for l in gls) >= 0.8 and size >= 0.95 * body and (text[0].isupper() or text[0].isdigit())
          and isolated(iso, 1.2)):
        b.label, b.why = "section_header", "emphasis_isolated"
    if b.label == "section_header":
        k = size_key(size)
        b.level = (ctx.heading_sizes.index(k) + 1) if k in ctx.heading_sizes else min(3, len(ctx.heading_sizes) + 1)
        return
    if starts_list(text):
        b.label, b.why = "list_item", "list_marker"
        return
    if len(gls) >= 2 and gls[0].fx0 < min(l.fx0 for l in gls[1:]) - 0.8 * size:
        b.label, b.why = "list_item", "hanging_indent"
        return


def picture_runs(blocks, glines, body):
    """Text inside a figure: the blocks next to a figure caption (before it and after it in the plain order) that
    are short (<= 2 rows and <= 100 chars) or mostly digits, stopping at prose (>= 3 rows), captions, tables,
    furniture and footnotes. A run needs >= 2 lines and (a mostly-digit line or >= 3 blocks)."""
    figure_like = ("paragraph", "section_header", "list_item", "formula", "title")
    for i, b in enumerate(blocks):
        if b.label != "caption" or not FIGWORD.match(b.text.translate(LOOKALIKE)):
            continue
        for step in (-1, 1):
            run, j = [], i + step
            while 0 <= j < len(blocks):
                o, gl = blocks[j], glines[j]
                if o.label not in figure_like or not gl:
                    break
                rows = sum(l.rows for l in gl)
                txt = o.text.strip()
                if not (rows <= 2 and len(txt) <= 100) and numeric_share(txt) < 0.5:
                    break
                if SECNUM.match(txt):
                    break
                run.append(j)
                j += step
            lines = [l for j in run for l in glines[j]]

            def figure_ish(j):            # small type, digits, or a stream that jumps around (axis labels)
                gl = glines[j]
                return (median_low([l.size for l in gl]) <= 0.95 * body or numeric_share(blocks[j].text) >= 0.5
                        or blocks[j].cut in ("not_below", "no_overlap", "angle") or blocks[j].rotated)
            if (len(lines) >= 2 and (len(run) >= 3 or any(numeric_share(l.text) >= 0.5 for l in lines))
                    and 2 * sum(figure_ish(j) for j in run) >= len(run)):
                for j in run:
                    blocks[j].label, blocks[j].level, blocks[j].why = "picture", None, "figure_text"


def list_neighbours(blocks, glines):
    """A one-row paragraph that starts at the first-line edge of an adjacent hanging-indent list item, at the same
    size, is an item of the same list (single-line references)."""
    for i, b in enumerate(blocks):
        gl = glines[i]
        if b.label != "paragraph" or not gl or sum(l.rows for l in gl) != 1:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(blocks) and blocks[j].why == "hanging_indent" and glines[j]:
                o = glines[j][0]
                if abs(o.fx0 - gl[0].fx0) <= 0.5 * o.size and abs(o.size - gl[0].size) <= SIZE_STEP * o.size:
                    b.label, b.why = "list_item", "list_neighbour"
                    break


def footnote_area(blocks, L, body, g):
    """The trailing run of small-text blocks (before the page footers) is the footnote area when one of its blocks
    starts with a note mark; the run must lie in the lower half of the page."""
    body_idx = [i for i, b in enumerate(blocks) if b.label not in ("page_header", "page_footer")]
    ref = body                                   # the size of the page's running paragraphs (>= 3 rows)
    for b in blocks:
        gl = [L[j] for j in range(b.first, b.last + 1) if L[j].geo]
        if b.label == "paragraph" and gl and sum(l.rows for l in gl) >= 3:
            ref = max(ref, median_low([l.size for l in gl]))
    body = ref
    run = []
    for i in reversed(body_idx):
        b = blocks[i]
        gl = [L[j] for j in range(b.first, b.last + 1) if L[j].geo]
        if not gl or b.label in ("table", "section_header"):
            break
        size = median_low([l.size for l in gl])
        if size > FOOT_SIZE * body or b.bbox is None or b.bbox[1] < 0.35 * g.H(gl[0]):
            break
        run.append(i)
    run.reverse()

    def marked(k):                      # a mark alone on its line reads with the next block of the run
        t = blocks[run[k]].text
        if re.fullmatch(r"\s*\S{1,3}\s*", t) and k + 1 < len(run):
            t = t + " " + blocks[run[k + 1]].text
        return bool(FOOTMARK.match(t)) and numeric_share(t) < 0.5
    low = run and blocks[run[-1]].bbox[3] >= 0.6 * g.H(L[blocks[run[-1]].first]) if run else False
    if run and low and any(marked(k) for k in range(len(run))):
        for i in run:
            if blocks[i].label in ("paragraph", "list_item"):
                blocks[i].label, blocks[i].why = "footnote", "footnote_area"


def title_rule(blocks, L, body, ctx, page_no, g):
    if page_no > 2:
        return
    cand = [b for b in blocks if b.label in ("section_header", "paragraph") and b.bbox]
    if not cand:
        return
    def bsize(b):
        return median_low([L[j].size for j in range(b.first, b.last + 1) if L[j].geo])
    big = max(cand, key=lambda b: (bsize(b), -b.first))
    if bsize(big) >= 1.5 * body and big.label == "section_header":
        big.label, big.level, big.why = "title", None, "title_first_pages"


def page_markdown(blocks):
    parts = []
    for b in blocks:
        if b.label == "title":
            parts.append("# " + b.text)
        elif b.label == "section_header":
            parts.append("#" * min(6, (b.level or 1) + 1) + " " + b.text)
        else:
            parts.append(b.text)
    return "\n\n".join(parts)
