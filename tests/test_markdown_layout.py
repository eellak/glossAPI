"""K5-A2: layout v2 Markdown wired into the safe extractor (glossapi.markdown_layout).

Covers: rendering rules and the text invariant; the production path (extractor -> Markdown + sidecar metadata);
text byte-identity of the one-load-per-page reader; the reader fallbacks (Python reader, pypdfium2 open/page
failures, stale native); the K5A1-G3 input contract; context selection and the incremental document pass; the
memory-budget reread; process-wide PDFium coordination (lock, static audit, overlap instrument with a negative
control, fork); the native API level; the mode-dependent extract contract.
"""
from __future__ import annotations

import ast
import json
import os
import re
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from glossapi import markdown_layout as ml
from glossapi.markdown_layout import reader as mlr
from glossapi.markdown_layout import render as mlrender
from layout_v2_ref import fuzz as FZ  # noqa: E402
from layout_v2_ref import k5chars as kc  # noqa: E402

native = mlr.native_module()[0]      # the extension module (maturin: glossapi_rs_pdfrepair.glossapi_rs_pdfrepair)
assert native is not None, "glossapi_rs_pdfrepair at native API level 3 is required: a stale build is first"
REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "samples/lightweight_pdf_corpus/pdfs/alpha.pdf"


@pytest.fixture(autouse=True)
def _fresh_native_state(monkeypatch):
    monkeypatch.delenv(ml.ENVIRONMENT, raising=False)
    mlr._reset_native_state()
    yield
    mlr._reset_native_state()


def _document_pdf(path, pages=4):
    """A Latin document with a title, numbered headings, paragraphs, a list, a caption, a table and page numbers."""
    from pdf_fixtures import FixturePdf
    fx = FixturePdf()
    para = ("A line of running text that goes all the way to the right margin of the page here.\n"
            "and it continues on the second line of the same paragraph with the same size.\n"
            "The last line.")
    for k in range(pages):
        page = fx.new_page(595, 842)
        if k == 0:
            fx.insert_text(page, (72, 70), "A Study of Layout Labels", fontsize=24)
        fx.insert_text(page, (72, 130), f"{k + 1}. Section number {k + 1}", fontsize=16)
        fx.insert_text(page, (72, 170), para, fontsize=10)
        fx.insert_text(page, (72, 230), para, fontsize=10)
        fx.insert_text(page, (72, 290), "- first point of the list\n- second point of the list", fontsize=10)
        fx.insert_text(page, (150, 340), f"Table {k + 1}: Results of the measurements", fontsize=9)
        fx.insert_text(page, (100, 360), "Alpha    10    20    30\nBeta    11    21    31\nGamma    12    22    32",
                       fontsize=10)
        fx.insert_text(page, (290, 800), str(k + 1), fontsize=10)
    return fx.save(path)


def _legacy_pages(path):
    """Production's page texts before K5-A2 (pypdfium2 get_text_range, newline normalisation, strip)."""
    import warnings
    import pypdfium2 as pdfium
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = pdfium.PdfDocument(str(path))
        try:
            for index in range(len(doc)):
                page = doc[index]
                tp = page.get_textpage()
                out.append(ml.page_text(tp.get_text_range()))
                tp.close()
                page.close()
        finally:
            doc.close()
    return out


def check_invariant(markdown, metadata, pages):
    """Rules D1/P1-P5 and the text invariant: every page entry is the page text (unlabelled) or the page text cut
    into blocks with only whitespace removed between them; markup between blocks is exactly the rule's prefix."""
    nonblank = [p for p in pages if p]
    assert [entry[0] for entry in metadata["pages"]] == [i + 1 for i, p in enumerate(pages) if p]
    assert markdown == "\n\n".join(markdown[s:e] for _, s, e, _ in metadata["pages"])
    blocks = {}
    for page, s, e, label, level in metadata["blocks"]:
        blocks.setdefault(page, []).append((s, e, label, level))
    for (page, start, end, status), plain in zip(metadata["pages"], nonblank):
        if status != "labelled":
            assert markdown[start:end] == plain and page not in blocks
            continue
        cursor, pos = start, 0
        for k, (s, e, label, level) in enumerate(blocks[page]):
            body = markdown[s:e]
            original = plain[pos:pos + len(body)]
            if label in ("title", "section_header"):
                assert original.replace("\n", " ") == body
            else:
                assert original == body
            prefix, rendered = mlrender.block_prefix(label, level, original)
            assert rendered == body
            assert markdown[cursor:s] == ("\n\n" if k else "") + prefix
            cursor, pos = e, pos + len(body)
            gap = len(plain[pos:]) - len(plain[pos:].lstrip())
            assert k + 1 == len(blocks[page]) or gap > 0       # blocks are separated by whitespace in the plain text
            pos += gap
        assert cursor == end and pos == len(plain)
    return True


# ---- rendering rules ---------------------------------------------------------------------------------------------
def test_rendering_rules_per_label():
    text = "Title\nrow two\n\n1.2 Head\nPara line\n• item\n- dash item\nFigure 1: x\n3 note\n12\nA 1 2\nPic\nx+y=z"
    cuts = [("Title\nrow two", "title", None), ("1.2 Head", "section_header", 2), ("Para line", "paragraph", None),
            ("• item", "list_item", None), ("- dash item", "list_item", None), ("Figure 1: x", "caption", None),
            ("3 note", "footnote", None), ("12", "page_footer", None), ("A 1 2", "table", None),
            ("Pic", "picture", None), ("x+y=z", "formula", None)]
    blocks, pos = [], 0
    for piece, label, level in cuts:
        start = text.index(piece, pos)
        blocks.append((start, start + len(piece), label, level))
        pos = start + len(piece)
    md, spans = mlrender.render_page(text, blocks)
    assert md == ("# Title row two\n\n### 1.2 Head\n\nPara line\n\n- • item\n\n- dash item\n\n"
                  "<!-- caption -->\nFigure 1: x\n\n<!-- footnote -->\n3 note\n\n<!-- page-footer -->\n12\n\n"
                  "<!-- table -->\nA 1 2\n\n<!-- picture -->\nPic\n\n<!-- formula -->\nx+y=z")
    assert [md[s:e] for s, e, _, _ in spans][:2] == ["Title row two", "1.2 Head"]
    assert mlrender.block_prefix("section_header", 9, "x")[0] == "###### "
    assert mlrender.block_prefix("page_header", None, "x")[0] == "<!-- page-header -->\n"
    assert mlrender.block_prefix("list_item", None, "12) item")[0] == ""
    assert set(mlrender.LABELS) == {"title", "section_header", "paragraph", "list_item", *mlrender.MARKS}


def test_mode_and_contract_version(monkeypatch):
    from glossapi.stages.extract import ExtractStage
    assert ml.mode() == "layout_v2" and ExtractStage().contract_version == 3   # K5-A2b: sidecar strip record
    monkeypatch.setenv(ml.ENVIRONMENT, "plain")
    assert ml.mode() == "plain" and ExtractStage().contract_version == 2
    monkeypatch.setenv(ml.ENVIRONMENT, "docling")
    with pytest.raises(ValueError, match="GLOSSAPI_MARKDOWN_LAYOUT"):
        ml.mode()


# ---- production path ---------------------------------------------------------------------------------------------
def _extract(path, out, monkeypatch, mode=None):
    from glossapi.gloss_extract import GlossExtract
    if mode:
        monkeypatch.setenv(ml.ENVIRONMENT, mode)
    else:
        monkeypatch.delenv(ml.ENVIRONMENT, raising=False)
    extractor = GlossExtract()
    extractor.use_pypdfium_backend = True
    extractor.repair_fonts = False
    extractor.pdf_backend_name = "pypdfium"
    md_dir = out / "markdown"
    md_dir.mkdir(parents=True)
    assert extractor._process_single_document_pypdfium(Path(path), output_dir=md_dir)
    sidecar = json.loads((out / "sidecars/extract" / f"{Path(path).stem}.json").read_text())
    return (md_dir / f"{Path(path).stem}.md").read_text(encoding="utf-8"), sidecar


def test_extractor_writes_labelled_markdown_with_unchanged_text(tmp_path, monkeypatch):
    pdf = _document_pdf(tmp_path / "doc.pdf")
    pages = _legacy_pages(pdf)
    md, sidecar = _extract(pdf, tmp_path / "layout", monkeypatch)
    meta = sidecar["markdown_layout"]
    assert meta["schema"] == "glossapi.markdown_layout.v1" and meta["rules"] == "glossapi-layout-md/1"
    assert meta["reader"] == "native" and meta["status"] == "labelled"
    assert check_invariant(md, meta, pages)
    labels = meta["label_counts"]
    assert labels.get("title", 0) >= 1 and labels.get("section_header", 0) >= 2 and labels.get("caption", 0) >= 4
    assert labels.get("page_footer", 0) >= 3 and md.startswith("# A Study of Layout Labels")
    assert meta["context"]["pages"] == 4 and meta["rejected"] == [] and meta["reread_pages"] == 0
    # plain mode: the Markdown and sidecar of extract contract 1, byte for byte
    plain_md, plain_sidecar = _extract(pdf, tmp_path / "plain", monkeypatch, "plain")
    assert plain_md == "\n\n".join(p for p in pages if p) and "markdown_layout" not in plain_sidecar
    assert {k: v for k, v in sidecar.items() if k != "markdown_layout"} == {k: v for k, v in plain_sidecar.items() if k != "plain_pages"}


def test_native_page_text_is_byte_identical_to_pypdfium2(tmp_path):
    pdf = _document_pdf(tmp_path / "doc.pdf", pages=40)       # more than one read chunk (32)
    for path in (pdf, SAMPLE, REPO / "samples/lightweight_pdf_corpus/pdfs/beta.pdf"):
        import warnings
        import pypdfium2 as pdfium
        source = ml.open_page_source(path, chars=True)
        assert source.kind == "native"
        doc = pdfium.PdfDocument(str(path))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in range(len(doc)):
                page = doc[index]
                tp = page.get_textpage()
                assert source.text(index) == tp.get_text_range()
                table = source.chars(index)
                x0, y0, x1, y1 = page.get_mediabox()
                import pypdfium2.raw as raw
                assert table == mlr.read_chars_python(raw, tp.raw, x1 - x0, y1 - y0, page.get_rotation(), index + 1)
                tp.close()
                page.close()
        doc.close()
        source.close()
    assert source.reads >= 1


def test_plain_mode_reads_text_only_through_the_native_reader(tmp_path, monkeypatch):
    pdf = _document_pdf(tmp_path / "doc.pdf", pages=3)
    calls = []
    real = native.layout_v2_read_document

    def recording(source, pages, sha, password=None, chars=True):
        calls.append(chars)
        return real(source, pages, sha, chars=chars)
    monkeypatch.setattr(native, "layout_v2_read_document", recording)
    md, sidecar = _extract(pdf, tmp_path / "p", monkeypatch, "plain")
    assert md == "\n\n".join(p for p in _legacy_pages(pdf) if p) and "markdown_layout" not in sidecar
    assert calls and not any(calls[1:])                     # page count call, then text-only chunk reads
    count, rows = real(str(pdf), [1, 2], bytes(32), chars=False)
    assert count == 3 and [r[1] for r in rows] == [None, None] and rows[0][2] == real(str(pdf), [1], bytes(32))[1][0][2]


def test_python_reader_fallback_gives_the_same_markdown(tmp_path, monkeypatch):
    pdf = _document_pdf(tmp_path / "doc.pdf")
    native_md, native_sidecar = _extract(pdf, tmp_path / "n", monkeypatch)

    def unavailable(*args, **kwargs):
        raise native.LayoutV2ReaderUnavailable("test: reader unavailable")
    monkeypatch.setattr(native, "layout_v2_read_document", unavailable)
    md, sidecar = _extract(pdf, tmp_path / "p", monkeypatch)
    assert md == native_md and sidecar["markdown_layout"]["reader"] == "python"
    meta = {k: v for k, v in sidecar["markdown_layout"].items() if k != "reader"}
    assert meta == {k: v for k, v in native_sidecar["markdown_layout"].items() if k != "reader"}


def test_unopenable_document_raises_exactly_what_pypdfium2_raises(tmp_path):
    import pypdfium2 as pdfium
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\nnot a pdf")
    with pytest.raises(Exception) as legacy:
        pdfium.PdfDocument(str(bad))
    with pytest.raises(type(legacy.value)) as ours:
        ml.open_page_source(bad, chars=True)
    assert str(ours.value) == str(legacy.value)


def test_native_page_error_reads_the_page_the_pypdfium2_way(tmp_path, monkeypatch):
    pdf = _document_pdf(tmp_path / "doc.pdf", pages=3)
    real = native.layout_v2_read_document

    def page_two_fails(source, pages, sha, password=None, chars=True):
        count, rows = real(source, pages, sha, chars=chars)
        return count, [(p, None, None, "injected") if p == 2 else (p, t, x, e) for p, t, x, e in rows]
    monkeypatch.setattr(native, "layout_v2_read_document", page_two_fails)
    md, sidecar = _extract(pdf, tmp_path / "x", monkeypatch)
    meta = sidecar["markdown_layout"]
    assert check_invariant(md, meta, _legacy_pages(pdf))
    assert [p[3] for p in meta["pages"]] == ["labelled", "no_char_table", "labelled"]


def test_stale_native_gives_plain_markdown_and_capabilities_report_it(tmp_path, monkeypatch):
    import types
    pdf = _document_pdf(tmp_path / "doc.pdf", pages=2)
    stale = types.ModuleType("glossapi_rs_pdfrepair")
    stale.__glossapi_native_api__ = "GLOSSAPI_NATIVE_API:glossapi_rs_pdfrepair:1;"   # a K5-A1r2-era build
    stale.layout_v2_read_document = native.layout_v2_read_document
    monkeypatch.setitem(sys.modules, "glossapi_rs_pdfrepair", stale)
    module, why = mlr.native_module()
    assert module is None and "stale native module" in why and ":3;" in why
    md, sidecar = _extract(pdf, tmp_path / "s", monkeypatch)
    assert md == "\n\n".join(p for p in _legacy_pages(pdf) if p)
    assert sidecar["markdown_layout"]["status"] == "unavailable" and "stale" in sidecar["markdown_layout"]["reason"]
    from glossapi.capabilities import NATIVE_API_LEVELS, _marker_level
    assert NATIVE_API_LEVELS["glossapi_rs_pdfrepair"] == 3
    assert _marker_level(b"x GLOSSAPI_NATIVE_API:glossapi_rs_pdfrepair:1; y", "glossapi_rs_pdfrepair") == 1


def test_native_declares_api_level_3_and_the_layout_exports():
    assert native.__glossapi_native_api__ == "GLOSSAPI_NATIVE_API:glossapi_rs_pdfrepair:3;"  # K5-A2b: decode status
    assert mlr.native_module()[0] is native


# ---- K5A1-G3 input contract --------------------------------------------------------------------------------------
def _table(size=10.0, page_w=595.0, angle=0.0):
    t = kc.CharTable(page_w, 842.0, 0, 1)
    t.fonts = ["Times"]
    for i, c in enumerate("Hello world"):
        box = (72.0 + 5 * i, 700.0, 77.0 + 5 * i, 710.0)
        t.ch.append(c); t.bbox.append(box); t.loose.append(box); t.size.append(size); t.weight.append(400)
        t.angle.append(angle); t.font.append(0); t.flags.append(0)
    return kc.dumps(t)


@pytest.mark.parametrize("kwargs, problem", [({"size": float("nan")}, "font size"), ({"size": float("inf")}, "font size"),
                                             ({"page_w": float("nan")}, "page box"), ({"page_w": 1e9}, "page box"),
                                             ({"angle": float("nan")}, "angle")])
def test_non_finite_inputs_are_rejected_and_the_page_stays_unlabelled(kwargs, problem):
    good, bad = _table(), _table(**kwargs)
    assert native.layout_v2_input_problem(good) is None
    assert problem in native.layout_v2_input_problem(bad)
    labeller = ml.DocumentLabeller(native)
    labeller.add(1, "Hello world", good)
    labeller.add(2, "Hello world", bad)
    doc = labeller.finish()
    assert [p.status for p in doc.pages] == ["labelled", "layout_input_rejected"]
    assert doc.rejected[0][0] == 2 and doc.context["pages"] == 1          # kept out of the document pass
    md, meta = ml.render_document(doc, reader="native")
    assert md.endswith("\n\nHello world") and meta["status"] == "partial" and meta["rejected"][0][0] == 2


def test_rejected_page_through_the_extractor_records_a_warning(tmp_path, monkeypatch):
    pdf = _document_pdf(tmp_path / "doc.pdf", pages=2)
    real = native.layout_v2_read_document
    nan_table = _table(size=float("nan"))

    def bad_sizes(source, pages, sha, password=None, chars=True):
        count, rows = real(source, pages, sha, chars=chars)
        return count, [(p, nan_table if p == 2 else t, x, e) for p, t, x, e in rows]
    monkeypatch.setattr(native, "layout_v2_read_document", bad_sizes)
    seen = []
    import glossapi.gloss_extract as ge
    monkeypatch.setattr(ge, "observe", lambda code, **kw: seen.append((code, kw)))
    md, sidecar = _extract(pdf, tmp_path / "r", monkeypatch)
    meta = sidecar["markdown_layout"]
    assert [p[3] for p in meta["pages"]] == ["labelled", "layout_input_rejected"]
    assert check_invariant(md, meta, _legacy_pages(pdf))
    assert any(code == "layout_input_rejected" and kw["page"] == 2 and "font size" in kw["problem"] for code, kw in seen)


# ---- context selection and the incremental document pass ----------------------------------------------------------
def test_context_selection_is_the_frozen_rule():
    assert ml.context_pages(1, [1]) == [1]
    assert ml.context_pages(5, [3]) == [1, 2, 3, 4, 5]
    assert ml.context_pages(100, [50]) == [1, 15, 29, 43, 47, 48, 49, 50, 51, 52, 53, 58, 72, 86, 100]
    assert ml.context_pages(20, range(1, 21)) == list(range(1, 21))


def test_builder_equals_the_one_shot_document_pass_on_fuzz_documents():
    def canon(c):
        return (repr(c.body), [repr(x) for x in c.heading_sizes], c.pages,
                [(f["band"], f["signature"], repr(f["y"]), f["pages"]) for f in c.furniture])
    docs = 0
    for doc in FZ.fuzz_documents(20261104, 30):
        tables = [(pg, kc.dumps(t)) for pg, t in doc["context"]]
        try:
            want = native.layout_v2_context(tables)
        except ValueError as error:
            builder = native.LayoutV2ContextBuilder()
            with pytest.raises(ValueError, match=re.escape(str(error).split(": ", 1)[-1])):
                for pg, data in tables:
                    builder.add(pg, data)
                builder.finish()
            continue
        builder = native.LayoutV2ContextBuilder()
        for pg, data in tables:
            builder.add(pg, data)
        assert builder.added == len(tables) and canon(builder.finish()) == canon(want)
        docs += 1
    assert docs >= 25


def test_labels_through_the_labeller_equal_direct_calls_and_reread_when_over_budget(tmp_path):
    pdf = _document_pdf(tmp_path / "doc.pdf", pages=6)
    kind, count, doc = ml.label_pdf(pdf)
    source = ml.open_page_source(pdf, chars=True)
    tables = {}
    for i in range(count):
        source.text(i)
        tables[i + 1] = source.chars(i)
    ctx = native.layout_v2_context(sorted(tables.items()))
    for page in doc.pages:
        direct = native.layout_v2_label_page(page.text, tables[page.page], ctx, page.page)
        assert (page.rotation, page.warnings, page.lines, page.blocks, page.markdown) == \
            (direct[0], list(direct[1]), list(direct[2]), list(direct[3]), direct[4])
    labeller = ml.DocumentLabeller(native, budget=0)
    for i in range(count):
        labeller.add(i + 1, ml.page_text(source.text(i)), source.chars(i))
    tight = labeller.finish(lambda page: source.reread_chars(page - 1))
    source.close()
    assert tight.reread_pages == count and [p.blocks for p in tight.pages] == [p.blocks for p in doc.pages]
    with pytest.raises(ValueError, match="ascending"):
        labeller.add(1, "x", None)


def test_targets_select_the_frozen_context(tmp_path):
    pdf = _document_pdf(tmp_path / "doc.pdf", pages=12)
    _, _, doc = ml.label_pdf(pdf, targets=[6])
    assert [p.page for p in doc.pages] == [6]
    assert doc.context["pages"] == len(ml.context_pages(12, [6]))


# ---- PDFium coordination (policy P7) -----------------------------------------------------------------------------
def _guarded(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                expr = item.context_expr
                if isinstance(expr, ast.Call) and getattr(expr.func, "id", getattr(expr.func, "attr", "")) == "pdfium_guard":
                    return True
    return False


def test_every_pypdfium2_document_in_the_package_is_opened_under_the_process_lock():
    sites, unguarded = 0, []
    for path in sorted((REPO / "src/glossapi").rglob("*.py")):
        if "scripts" in path.parts:
            continue                                   # benchmarks and maintenance scripts: separate processes
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "PdfDocument":
                sites += 1
                if not _guarded(node, parents):
                    unguarded.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert sites >= 9 and not unguarded, unguarded


def test_nothing_in_the_package_or_the_reader_initializes_or_destroys_pdfium():
    pattern = re.compile(r"FPDF_InitLibrary|FPDF_DestroyLibrary|init_lib\(|destroy_lib\(")
    hits = [str(p.relative_to(REPO)) for root in ("src/glossapi", "rust/glossapi_rs_pdfrepair/src")
            for p in sorted((REPO / root).rglob("*")) if p.suffix in (".py", ".rs") and pattern.search(p.read_text())]
    assert hits == []
    functions = native.layout_v2_reader_info()["functions"]
    assert len(functions) == 24 and not [f for f in functions if "Library" in f]


OVERLAP = textwrap.dedent(r'''
import sys, threading, time, json, warnings
warnings.simplefilter("ignore")
from pathlib import Path
import pypdfium2 as pdfium
import pypdfium2.raw as raw
from glossapi import markdown_layout as ml
native = ml.native_module()[0]
from glossapi.page_quality import page_quality
pdf, guarded = Path(sys.argv[1]), sys.argv[2] == "guarded"
native.layout_v2_reader_info()                    # resolve the native reader before the raw functions are wrapped
state = {"inflight": 0, "overlaps": 0, "native_calls": 0, "ctypes_calls": 0}
def wrap(name, fn):
    def call(*args):
        state["inflight"] += 1; state["ctypes_calls"] += 1
        try:
            time.sleep(0.0002)                    # a long PDFium call: the GIL is free meanwhile
            return fn(*args)
        finally:
            state["inflight"] -= 1
    return call
import ctypes
for name in dir(raw):
    if name.startswith(("FPDF_", "FPDFText_", "FPDFPage_")) and isinstance(getattr(raw, name), ctypes._CFuncPtr):
        setattr(raw, name, wrap(name, getattr(raw, name)))
read = native.layout_v2_read_document
def counted(*args, **kwargs):
    state["native_calls"] += 1
    if state["inflight"]:
        state["overlaps"] += 1
    return read(*args, **kwargs)
native.layout_v2_read_document = counted
def pypdfium2_user():
    for _ in range(25):
        if guarded:
            page_quality(pdf)                     # a glossAPI PDFium user (guarded)
        else:
            doc = pdfium.PdfDocument(str(pdf))     # an unguarded user: the negative control
            for i in range(len(doc)):
                p = doc[i]; tp = p.get_textpage(); tp.get_text_range(); tp.close(); p.close()
            doc.close()
def native_user():
    for _ in range(25):
        ml.label_pdf(pdf)
errors = []
def run(fn):
    try:
        fn()
    except BaseException as error:
        errors.append(repr(error))
threads = [threading.Thread(target=run, args=(pypdfium2_user,)), threading.Thread(target=run, args=(native_user,))]
[t.start() for t in threads]; [t.join() for t in threads]
state["errors"] = errors
print(json.dumps(state))
''')


@pytest.mark.parametrize("guarded", [True, False])
def test_native_reads_never_overlap_package_pypdfium2_calls(tmp_path, guarded):
    """With the lock (glossAPI's own users) no native read starts while a PDFium call is in flight in another thread;
    the unguarded control shows that the instrument sees overlaps when the lock is not used. The control breaks the
    coordination contract on purpose: PDFium may also crash then (seen on Linux: SIGSEGV), which equally shows that
    uncoordinated use is unsafe; the guarded run must always exit 0."""
    pdf = _document_pdf(tmp_path / "doc.pdf", pages=3)
    done = subprocess.run([sys.executable, "-c", OVERLAP, str(pdf), "guarded" if guarded else "control"],
                          capture_output=True, text=True, timeout=600, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    if not guarded and done.returncode < 0:
        return                                            # the control crashed inside PDFium: uncoordinated use
    assert done.returncode == 0, done.stderr[-2000:]
    state = json.loads(done.stdout.strip().splitlines()[-1])
    assert state["errors"] == [] and state["inflight"] == 0, state
    assert state["native_calls"] >= 50 and state["ctypes_calls"] > 300, state
    if guarded:
        assert state["overlaps"] == 0, state
    else:
        assert state["overlaps"] > 0, state


def test_forked_child_gets_an_unlocked_guard():
    code = textwrap.dedent('''
        import os, threading, time
        from glossapi.pdfium_guard import pdfium_guard
        held, release = threading.Event(), threading.Event()
        def holder():
            with pdfium_guard():
                held.set(); release.wait()
        t = threading.Thread(target=holder); t.start(); held.wait()
        pid = os.fork()
        if pid == 0:
            ok = pdfium_guard().acquire(timeout=5)
            os._exit(0 if ok else 3)
        _, status = os.waitpid(pid, 0)
        release.set(); t.join()
        print(os.waitstatus_to_exitcode(status))
    ''')
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "0", done.stderr


def test_threat_model_statement_is_in_the_module_docs():
    statement = ("Documents and public argument values are untrusted; the Python runtime, installed dependencies and "
                 "executable caller code are trusted.")
    rust = (REPO / "rust/glossapi_rs_pdfrepair/src/layout_reader_py.rs").read_text()
    flat = " ".join(line.lstrip("/! ").strip() for line in rust.splitlines())
    assert statement in flat and "guarantee unlimited resource availability" in flat
    assert statement in " ".join(mlr.__doc__.split())
    assert statement in " ".join((REPO / "docs/stages/extract.md").read_text().split())
