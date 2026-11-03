"""K5-A1: the Rust port of layout v2 (glossapi_rs_pdfrepair.layout_v2_*) equals the frozen Python reference.

The reference (`tests/layout_v2_ref/labels_v2.py`, byte copy of the K5g freeze) and the port get the same input: the
page's plain text and its `.k5chars` char table (bytes for Rust, the round-tripped table for Python). Compared per page:
line pieces, every block (offsets, first/last line, label, level, rotated, why, cut, bbox), warnings, rotation and
the markdown string; per document: body size, heading sizes and the furniture list. Floats are compared by repr.

Also here: the Python-semantics primitives over every code point, the native PDFium char reader against the
reference reader, bad input, a huge page, concurrent calls, and the abi3 extension on this interpreter. The real-page
differential (98 tuning + 57 held-out pages) and the >= 1,000-page fuzz run live with the K5-A1 receipts, because
the page fixtures are not part of the repository. Nothing in `src/` uses the layout yet.
"""
from __future__ import annotations

import _sre
import ctypes
import hashlib
import math
import random
import re
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

native = pytest.importorskip("glossapi_rs_pdfrepair")

from layout_v2_ref import fuzz as FZ  # noqa: E402
from layout_v2_ref import k5chars as kc  # noqa: E402
from layout_v2_ref import labels_v2 as lv  # noqa: E402

REF_SHA256 = {"labels_v2.py": "23d5bae30db4a95bae993a31a3209ad4b715133c76b69cbf217647aff22aa850",
              "k5chars.py": "42c22df1b256ac4d29cbaa9e9fb5771a2042b6a8eb7e1313b829cb1aa2a3f7da"}


@pytest.fixture(autouse=True)
def _require_layout_api():
    # A stale extension built before K5-A1 lacks the port; this module tests the current build.
    assert hasattr(native, "layout_v2_label_page"), "glossapi_rs_pdfrepair lacks layout_v2: a stale build is first"


def r(x):
    return repr(x) if isinstance(x, float) else x


def ctx_py(tables):
    return lv.document_context({pg: lv.page_geo(lv.fixture_text(t), t) for pg, t in tables})


def ctx_rs(tables):
    return native.layout_v2_context([(pg, kc.dumps(t)) for pg, t in tables])


def canon_ctx(c):
    return (r(c.body), [r(x) for x in c.heading_sizes], c.pages,
            [(f["band"], f["signature"], r(f["y"]), f["pages"]) for f in c.furniture])


def page_py(plain, t, ctx, page_no):
    g = lv.page_geo(plain, t)
    res = lv.label_page(plain, t, ctx, page_no, g)
    blocks = [(g.lines[b.first].start, g.lines[b.last].end, b.first, b.last, b.label, b.level, b.rotated, b.why, b.cut,
               None if b.bbox is None else tuple(r(v) for v in b.bbox)) for b in res.blocks]
    return (res.rotation, res.warnings, [(ln.start, ln.end) for ln in g.lines], blocks, lv.page_markdown(res.blocks))


def page_rs(plain, data, ctx, page_no):
    rot, warnings, lines, blocks, md = native.layout_v2_label_page(plain, data, ctx, page_no)
    out = [(s, e, first, last, label, level, rotated, why, cut, None if bbox is None else tuple(r(v) for v in bbox))
           for (s, e, label, level, rotated, why, cut, first, last, bbox) in blocks]
    return (rot, warnings, [tuple(x) for x in lines], out, md)


def assert_document_equal(tables, pages):
    """tables: [(page, CharTable)] context; pages: [(page, CharTable, plain)]."""
    cp, cr = ctx_py(tables), ctx_rs(tables)
    assert canon_ctx(cp) == canon_ctx(cr)
    for pno, t, plain in pages:
        want = page_py(plain, t, cp, pno)
        got = page_rs(plain, kc.dumps(t), cr, pno)
        assert got == want, (pno, plain[:80])
        # the invariant: block texts + the original separators == plain, separators are whitespace
        blocks = got[3]
        if blocks:
            assert blocks[0][0] == 0 and blocks[-1][1] == len(plain)
            for a, b in zip(blocks, blocks[1:]):
                assert plain[a[1]:b[0]].isspace()
    return len(pages)


# ---- reference ---------------------------------------------------------------------------------------------------
def test_reference_is_the_frozen_copy():
    here = Path(lv.__file__).parent
    for name, sha in REF_SHA256.items():
        assert hashlib.sha256((here / name).read_bytes()).hexdigest() == sha, name


# ---- synthetic rules (the K5g unit cases, run on both sides) -----------------------------------------------------
def make_table(rows, page_w=595.0, page_h=842.0, rotation=0):
    """rows: (text, x, top, size, font, angle_deg); the K5g test helper."""
    t = kc.CharTable(page_w, page_h, rotation, 1)
    fonts = {}
    for k, (text, x, top, size, font, ang) in enumerate(rows):
        fid = fonts.setdefault(font, len(fonts))
        cx = x
        for c in text:
            w = 0.5 * size
            box = (cx, page_h - top - size, cx + w, page_h - top) if ang == 0 else (top, cx, top + size, cx + w)
            t.ch.append(c)
            t.bbox.append(box)
            t.loose.append(box)
            t.size.append(size)
            t.weight.append(400)
            t.angle.append(math.radians(ang))
            t.font.append(fid)
            t.flags.append(0)
            cx += w
        if k + 1 < len(rows):
            for c in "\r\n":
                t.ch.append(c)
                t.bbox.append((0.0, 0.0, 0.0, 0.0))
                t.loose.append((0.0, 0.0, 0.0, 0.0))
                t.size.append(0.0)
                t.weight.append(0)
                t.angle.append(0.0)
                t.font.append(0)
                t.flags.append(lv.GENERATED)
    t.fonts = [f for f, _ in sorted(fonts.items(), key=lambda kv: kv[1])]
    return kc.loads(kc.dumps(t))


PARA = ["Αυτή είναι μια γραμμή κειμένου που φτάνει μέχρι το δεξί περιθώριο της σελίδας εδώ.",
        "και συνεχίζει στη δεύτερη γραμμή της ίδιας παραγράφου με το ίδιο μέγεθος και ύφος.",
        "Τελευταία γραμμή."]


def body(top, lines=PARA, x=72.0, size=10.0):
    return [(s, x, top + 12.0 * i, size, "Times", 0) for i, s in enumerate(lines)]


SYNTHETIC = {
    "paragraphs": (body(100) + body(150), ["paragraph", "paragraph"]),
    "heading_big": ([("1. Εισαγωγή στο θέμα", 72, 70, 16.0, "Times-Bold", 0)] + body(100),
                    ["section_header", "paragraph"]),
    "numbered_isolated": (body(100) + [("2.1. Συλλογή δεδομένων", 72, 160, 10.0, "Times", 0)] + body(185),
                          ["paragraph", "section_header", "paragraph"]),
    "list": (body(100, ["• πρώτο σημείο της λίστας", "• δεύτερο σημείο της λίστας"]), ["list_item", "list_item"]),
    "caption": (body(100) + [("Πίνακας 3: Αποτελέσματα μετρήσεων", 150, 170, 9.0, "Times", 0)], None),
    "footnote_and_page_number": (body(100) + body(150) + [
        ("12 Βλ. Παπαδόπουλος 2001, σελ. 4 και τα σχετικά σχόλια στην υποσημείωση.", 72, 700, 8.0, "Times", 0),
        ("57", 290, 800, 10.0, "Times", 0)], None),
    "table": (body(100) + [("Α    10    20    30", 100, 160, 10.0, "Times", 0),
                           ("Β    11    21    31", 100, 172, 10.0, "Times", 0),
                           ("Γ    12    22    32", 100, 184, 10.0, "Times", 0)], None),
    "rotated": ([("Περιστραμμένο κείμενο πίνακα σε οριζόντια σελίδα", 100, 72 + 12 * k, 10.0, "Times", 270)
                 for k in range(3)], ["paragraph"]),
}


@pytest.mark.parametrize("name", sorted(SYNTHETIC))
def test_synthetic_rules_equal_reference(name):
    rows, labels = SYNTHETIC[name]
    t = make_table(rows)
    plain = lv.fixture_text(t)
    assert_document_equal([(5, t)], [(5, t, plain)])
    got = [b[2] for b in native.layout_v2_label_page(plain, kc.dumps(t), ctx_rs([(5, t)]), 5)[3]]
    if labels:
        assert got == labels
    expect_last = {"caption": "caption", "table": "table", "footnote_and_page_number": "page_footer"}
    if name in expect_last:
        assert got[-1] == expect_last[name]


def test_rows_joined_without_newline_split_only_at_whitespace():
    t = make_table([("πρώτη γραμμή", 72, 100, 10.0, "Times", 0), ("δεύτερη", 72, 140, 10.0, "Times", 0)])
    k = t.ch.index("\r")
    t.ch[k:k + 2] = [" "]
    for f in ("bbox", "loose", "size", "weight", "angle", "font", "flags"):
        arr = getattr(t, f)
        arr[k:k + 2] = [arr[k]]
    plain = lv.fixture_text(t)
    assert plain == "πρώτη γραμμή δεύτερη"
    assert_document_equal([(5, t)], [(5, t, plain)])
    _, _, lines, blocks, _ = native.layout_v2_label_page(plain, kc.dumps(t), ctx_rs([(5, t)]), 5)
    assert [plain[a:b] for a, b in lines] == ["πρώτη γραμμή", "δεύτερη"] and len(blocks) == 2


# ---- seeded fuzz ---------------------------------------------------------------------------------------------------
def test_seeded_fuzz_documents_equal_reference():
    pages = 0
    for doc in FZ.fuzz_documents(20261103, 24):
        pages += assert_document_equal(doc["context"], doc["pages"])
    assert pages >= 120


# ---- Python-semantics primitives -----------------------------------------------------------------------------------
def test_char_predicates_and_re_lower_on_every_code_point():
    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    funcs = {"isspace": str.isspace, "isalpha": str.isalpha, "isdecimal": str.isdecimal, "isdigit": str.isdigit,
             "isnumeric": str.isnumeric, "isalnum": str.isalnum, "isupper": str.isupper}
    for k in range(0, len(cps), 100000):
        chunk = "".join(map(chr, cps[k:k + 100000]))
        for name, f in funcs.items():
            assert native._layout_v2_probe(name, chunk) == [f(c) for c in chunk], name
        assert native._layout_v2_probe("re_lower", chunk) == [_sre.unicode_tolower(ord(c)) for c in chunk]
        joined = "\x00".join(chunk)
        assert native._layout_v2_probe("nfc", joined) == "\x00".join(unicodedata.normalize("NFC", c) for c in chunk)
        assert native._layout_v2_probe("casefold", joined) == "\x00".join(c.casefold() for c in chunk)


PATTERNS = {"BOLD_NAME": ("search", lv.BOLD_NAME), "ITALIC_NAME": ("search", lv.ITALIC_NAME),
            "MATH_FONT": ("search", lv.MATH_FONT), "BULLET": ("match", lv.BULLET), "ENUM": ("match", lv.ENUM),
            "NUMBERED": ("match", lv.NUMBERED), "SECNUM": ("match", lv.SECNUM), "CAPTION": ("match", lv.CAPTION),
            "FOOTMARK": ("match", lv.FOOTMARK), "PAGE_NUM": ("match", lv.PAGE_NUM),
            "LINE_START": ("match", lv.LINE_START), "FIGWORD": ("match", lv.FIGWORD),
            "TABLE_RULE": ("match", re.compile(r"^\s*\|?\s*:?-{3,}")),
            "MARK_ALONE": ("fullmatch", re.compile(r"\s*\S{1,3}\s*"))}
PIECES = ["bold", "-b", ",b", "-bd", "cmbx", "grx", "bx", "italic", "-it", "cmmi", "symbol", "mt extra", "math",
          "1.", "2.1.", "IV.", "ΙV.1", "Α.1", "(a)", "α)", "o ", "• ", "–", "", "* ", "#", "|", "---", "|:--",
          "Σχήμα", "Πίνακας", "Πιν.", "∆ιάγραµµα", "Figure", "Fig.", "Κεφάλαιο", "Chapter", "¹", "†", "12", "1234",
          "xiv", "αβγ΄", "ιβ'", "ʹ", "[", "]", "(", ")", "-", "—", "·", ".", " ", "\t", "\n", "5 α .", "A. B", "F.A.B.",
          "ſ", "ı", "İ", "K", "ϐ", "ϵ", "ϑ", "ϰ", "ϖ", "ϱ", "ϕ", "µ", "ͅ", "ι", "Ω", "Σ", "ς", "\x1c",
          "　", "٣", "²", "x"]


def test_patterns_follow_python_re():
    rng = random.Random(5)
    for name, (mode, pat) in PATTERNS.items():
        f = getattr(pat, mode)
        for _ in range(3000):
            s = "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 6)))
            if rng.random() < 0.3:
                s = s.upper() if rng.random() < 0.5 else s.swapcase()
            assert native._layout_v2_probe(name, s) == (f(s) is not None), (name, s)


def test_norm_sig_and_round2():
    rng = random.Random(6)
    alphabet = "αάΑΆσςΣ́̈ͅᾳ가각eE0123٣²½  \t　ßİſ"
    for _ in range(5000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 16)))
        assert native._layout_v2_probe("norm_sig", s) == lv.norm_sig(s), s
    for _ in range(20000):
        x = rng.choice([rng.randint(-99999, 99999) / 64, rng.randint(-99999, 99999) / 1000 + 0.0005,
                        rng.uniform(-1e4, 1e4), rng.uniform(-1, 1) * 10.0 ** rng.randint(-9, 18)])
        assert repr(native._layout_v2_round2(x)) == repr(round(x, 2)), x


# ---- native PDFium reader ------------------------------------------------------------------------------------------
FLAG_SYMBOLIC, FLAG_ITALIC = 4, 64


def read_chars_reference(raw, tp, page_w, page_h, rotation, page_no, sha):
    """The K5c fixture reader (k5_prep/layout/freeze_chars.read_chars), copied."""
    n = raw.FPDFText_CountChars(tp)
    t = kc.CharTable(page_w, page_h, rotation, page_no, sha)
    fonts = {}
    l, rr, b, tt = (ctypes.c_double() for _ in range(4))
    rect = raw.FS_RECTF()
    fl = ctypes.c_int()
    buf = ctypes.create_string_buffer(512)
    for i in range(n):
        u = raw.FPDFText_GetUnicode(tp, i)
        raw.FPDFText_GetCharBox(tp, i, ctypes.byref(l), ctypes.byref(rr), ctypes.byref(b), ctypes.byref(tt))
        raw.FPDFText_GetLooseCharBox(tp, i, ctypes.byref(rect))
        size = raw.FPDFText_GetFontSize(tp, i)
        weight = raw.FPDFText_GetFontWeight(tp, i)
        ln = raw.FPDFText_GetFontInfo(tp, i, buf, len(buf), ctypes.byref(fl))
        name = buf.raw[:max(0, ln - 1)].decode("utf-8", "replace") if ln > 0 else ""
        angle = raw.FPDFText_GetCharAngle(tp, i)
        f = 0
        f |= kc.GENERATED if raw.FPDFText_IsGenerated(tp, i) == 1 else 0
        f |= kc.SOFT_HYPHEN if u == 0x02 else 0
        f |= kc.MAP_ERROR if raw.FPDFText_HasUnicodeMapError(tp, i) == 1 else 0
        f |= kc.ITALIC if ln > 0 and fl.value & FLAG_ITALIC else 0
        f |= kc.SYMBOLIC if ln > 0 and fl.value & FLAG_SYMBOLIC else 0
        f |= kc.INVISIBLE if raw.FPDFText_GetTextRenderMode(tp, i) == 3 else 0
        try:
            c = chr(u)
        except (ValueError, OverflowError):
            c = "�"
        t.ch.append(c)
        t.bbox.append((l.value, b.value, rr.value, tt.value))
        t.loose.append((rect.left, rect.bottom, rect.right, rect.top))
        t.size.append(float(size))
        t.weight.append(max(-32768, min(32767, int(weight))))
        t.angle.append(float(angle))
        t.font.append(fonts.setdefault(name, len(fonts)))
        t.flags.append(f)
    t.fonts = list(fonts)
    return t


def _pdfium_functions(raw):
    return [ctypes.cast(getattr(raw, name), ctypes.c_void_p).value for name in native.layout_v2_pdfium_functions()]


def test_native_reader_equals_reference_reader(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    import pypdfium2.raw as raw
    from pdf_fixtures import FixturePdf
    fx = FixturePdf()
    for k, (w, h, rot) in enumerate([(595, 842, 0), (842, 595, 90), (612, 792, 270)]):
        page = fx.new_page(w, h)
        fx.insert_text(page, (72, 80), "1. Introduction to the subject", fontsize=16)
        fx.insert_text(page, (72, 120), "A line of running text that goes to the margin.\nThe second line.\n"
                       "Table 1: Results    10    20    30", fontsize=10)
        fx.insert_text(page, (290, 800 if h > 800 else 580), str(k + 1), fontsize=10)
        if rot:
            page.set_rotation(rot)
    path = fx.save(tmp_path / "reader.pdf")
    repo = Path(__file__).resolve().parents[1]
    sha = hashlib.sha256(path.read_bytes()).digest()
    fns = _pdfium_functions(raw)
    tables = []
    for pdf in (path, repo / "samples/lightweight_pdf_corpus/pdfs/alpha.pdf"):
        doc = pdfium.PdfDocument(str(pdf))
        try:
            for i in range(len(doc)):
                page = doc[i]
                tp = page.get_textpage()
                try:
                    x0, y0, x1, y1 = page.get_mediabox()
                    args = (x1 - x0, y1 - y0, page.get_rotation(), i + 1)
                    want = kc.dumps(read_chars_reference(raw, tp.raw, *args, sha))
                    got = native.layout_v2_read_textpage(fns, ctypes.cast(tp.raw, ctypes.c_void_p).value, *args, sha)
                    assert got == want, (pdf.name, i)
                    tables.append((i + 1, kc.loads(got), tp.get_text_bounded()))
                finally:
                    tp.close()
                    page.close()
        finally:
            doc.close()
    assert sum(len(t.ch) for _, t, _ in tables) > 100
    pages = [(p, t, text.replace("\r\n", "\n").replace("\r", "\n").strip()) for p, t, text in tables[:3]]
    assert_document_equal([(p, t) for p, t, _ in pages], pages)


def test_native_reader_refuses_bad_handles():
    raw = pytest.importorskip("pypdfium2.raw")
    fns = _pdfium_functions(raw)
    with pytest.raises(ValueError):
        native.layout_v2_read_textpage(fns[:5], 1, 595.0, 842.0, 0, 1, b"\0" * 32)
    with pytest.raises(ValueError):
        native.layout_v2_read_textpage([0] * 11, 1, 595.0, 842.0, 0, 1, b"\0" * 32)
    with pytest.raises(ValueError):
        native.layout_v2_read_textpage(fns, 0, 595.0, 842.0, 0, 1, b"\0" * 32)
    with pytest.raises(ValueError):
        native.layout_v2_read_textpage(fns, 1, 595.0, 842.0, 0, 1, b"short")


# ---- crash and lifecycle -------------------------------------------------------------------------------------------
def test_bad_input_raises_and_never_crashes():
    t = make_table(body(100))
    good = kc.dumps(t)
    ctx = ctx_rs([(1, t)])
    bad_font = bytearray(good)
    off = kc._HEAD.size + 2
    bad_font[off] = 0xFF                                   # font name no longer UTF-8
    over = bytearray(good)
    first_ch = off + len("Times".encode())
    over[first_ch:first_ch + 4] = (0x110000).to_bytes(4, "little")
    for data in (b"", b"K5CHARS\0", good[:-1], good + b"\0", bytes(bad_font), bytes(over), b"\xff" * 4096,
                 good[:12] + b"\xff\xff\xff\x7f" + good[16:]):
        with pytest.raises(ValueError):
            native.layout_v2_label_page("x", data, ctx, 1)
        with pytest.raises(ValueError):
            native.layout_v2_context([(1, data)])
    with pytest.raises(TypeError):
        native.layout_v2_label_page("x", "not bytes", ctx, 1)
    with pytest.raises(TypeError):
        native.layout_v2_label_page("x", good, object(), 1)
    nan = kc.loads(good)
    nan.angle[0] = float("nan")
    with pytest.raises(ValueError):
        native.layout_v2_label_page(lv.fixture_text(nan), kc.dumps(nan), ctx, 1)
    with pytest.raises(ValueError):
        lv.label_page(lv.fixture_text(nan), nan, ctx_py([(1, t)]), 1)    # the reference raises there too
    for w in (float("inf"), float("nan"), 1e300):
        odd = kc.loads(good)
        odd.page_w = w
        with pytest.raises(ValueError):                    # refused by design (reply.md): no finite page box
            native.layout_v2_label_page("x", kc.dumps(odd), ctx, 1)
    with pytest.raises(UnicodeEncodeError):                # a str with a lone surrogate cannot cross into Rust
        native.layout_v2_label_page("a\ud800b", good, ctx, 1)
    assert native.layout_v2_label_page("", kc.dumps(make_table([])), ctx, 1)[3] == []


def test_huge_page_is_labelled_without_overflow():
    rng = random.Random(9)
    rows = []
    for k in range(3000):
        rows.append(("".join(rng.choice("αβγδε ζηθ") for _ in range(90)), 40.0, 20.0 + 10.0 * k, 8.0, "Times", 0))
    # plus one 250,000-char line that starts with 50,000 spaces (nested \s* patterns, long repetitions)
    t = make_table(rows + [(" " * 50000 + "•" + "x" * 200000, 10.0, 30100.0, 0.01, "Times", 0)], page_h=30500.0)
    plain = lv.fixture_text(t)
    data = kc.dumps(t)
    t0 = time.perf_counter()
    ctx = native.layout_v2_context([(1, data)])
    _, _, lines, blocks, md = native.layout_v2_label_page(plain, data, ctx, 1)
    assert time.perf_counter() - t0 < 10
    assert len(lines) == 3001 and blocks and blocks[0][0] == 0 and blocks[-1][1] == len(plain)
    assert "".join(plain[a:b] + (plain[b:c] if c is not None else "") for (a, b, *_), c in
                   zip(blocks, [x[0] for x in blocks[1:]] + [None])) == plain


def test_concurrent_calls_equal_sequential_calls():
    docs = list(FZ.fuzz_documents(77, 6, edge=False))
    jobs = []
    for doc in docs:
        ctx = ctx_rs(doc["context"])
        jobs += [(plain, kc.dumps(t), ctx, p) for p, t, plain in doc["pages"]]
    want = [native.layout_v2_label_page(*j) for j in jobs]
    with ThreadPoolExecutor(8) as ex:
        for _ in range(3):
            assert list(ex.map(lambda j: native.layout_v2_label_page(*j), jobs)) == want
        tables = [[(p, kc.dumps(t)) for p, t in d["context"]] for d in docs]
        ctxs = list(ex.map(native.layout_v2_context, tables * 3))
        assert [canon_ctx(c) for c in ctxs] == [canon_ctx(c) for c in [ctx_rs(d["context"]) for d in docs] * 3]


def test_abi3_extension_imports_on_this_interpreter():
    code = ("import glossapi_rs_pdfrepair as m, glob, os; "
            "d = os.path.dirname(m.__file__); "
            "print(sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, '*.so')) + glob.glob(os.path.join(d, '*.pyd')))); "
            "print(m.__glossapi_native_api__); print(hasattr(m, 'layout_v2_label_page'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.splitlines()
    assert any("abi3" in name for name in eval(out[0])), out[0]       # noqa: S307 - our own repr of a list
    assert out[1] == "GLOSSAPI_NATIVE_API:glossapi_rs_pdfrepair:1;"     # the package's required level is unchanged
    assert out[2] == "True"
