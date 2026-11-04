"""K5-A2b: cleaner metrics on the markup-free text (K5A2A-G2) and the OCR routing policy items P4a, P4b, P4d.

Covers: the strip record of the layout Markdown (``markdown_layout.render``: edits that undo the markup) and the one
shared strip function ``markup_free``; the extractor's record; cleaner metrics and decision columns equal between the
default (layout) and the plain Markdown mode through the public API (stage adapters), the legacy subprocess profile
and ``Corpus.run``, with a control showing the comparison is sensitive; the score-only pass after OCR with an appended
supplement; P4a (all-GLYPH text -> OCR) with the cleaner's own glyph matchers; P4b (per-page ``decode_status`` from the
native content-stream check: the K5P-093 derivation -> partial, a valid page -> complete, a check that cannot run ->
unknown; sidecar and ledger page coverage); P4d (a flagged document whose page quality routes no page is not OCRed by
``Corpus.ocr``, as by ``run``; ``math_only`` unchanged).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from glossapi import markdown_layout as ml
from glossapi.markdown_layout import reader as mlr
from glossapi.markdown_layout import render as mlrender

METRICS = ("mojibake_badness_score", "latin_percentage", "percentage_greek", "char_count_no_comments", "is_empty",
           "greek_badness_score", "polytonic_ratio", "latin_word_shape_rate", "pua_char_count", "control_char_count")
DECISIONS = ("filter", "needs_ocr", "extract_status", "ocr_pages", "ocr_pages_count")


@pytest.fixture(autouse=True)
def _fresh_native_state(monkeypatch):
    monkeypatch.delenv(ml.ENVIRONMENT, raising=False)
    mlr._reset_native_state()
    yield
    mlr._reset_native_state()


# ---- the strip record and markup_free ------------------------------------------------------------------------------
def _page(text, cuts):
    blocks, pos = [], 0
    for piece, label, level in cuts:
        start = text.index(piece, pos)
        blocks.append((start, start + len(piece), label, level))
        pos = start + len(piece)
    return blocks


def test_edits_undo_every_rule_and_separator():
    text = ("Title\nrow two\n1.2 Head\nover two\n\nPara line • item\n- dash item\nFigure 1: x 3 note\n12\n"
            "A 1 2\nPic\nx+y=z")
    cuts = [("Title\nrow two", "title", None), ("1.2 Head\nover two", "section_header", 2),
            ("Para line", "paragraph", None), ("• item", "list_item", None), ("- dash item", "list_item", None),
            ("Figure 1: x", "caption", None), ("3 note", "footnote", None), ("12", "page_footer", None),
            ("A 1 2", "table", None), ("Pic", "picture", None), ("x+y=z", "formula", None)]
    blocks = _page(text, cuts)
    md, spans, edits = mlrender.render_page_with_edits(text, blocks)
    assert (md, spans) == mlrender.render_page(text, blocks)
    assert md.startswith("# Title row two\n\n### 1.2 Head over two\n\nPara line\n\n- • item")
    assert mlrender.apply_edits(md, edits) == text
    # leading/trailing text outside the blocks comes back too
    blocks = [(2, 6, "section_header", 1), (7, 9, "paragraph", None)]
    md, _, edits = mlrender.render_page_with_edits("  Head\nxy tail", blocks)
    assert md == "## Head\n\nxy" and mlrender.apply_edits(md, edits) == "  Head\nxy tail"
    with pytest.raises(ValueError):
        mlrender.apply_edits("abc", [[2, 3, ""], [0, 1, ""]])


def _record(md, edits, plain):
    return {"strip": mlrender.strip_record(md, edits, plain)}


def test_markup_free_checks_what_it_strips():
    text = "Head\nbody one\nbody two"
    md, _, edits = mlrender.render_page_with_edits(text, [(0, 4, "section_header", 1), (5, 22, "paragraph", None)])
    layout = _record(md, edits, text)
    assert ml.markup_free(md, layout) == text
    supplement = "\n\n<!-- ocr-pages: 2 -->\n<!-- ocr-supplement-begin: x -->\nOCR text\n<!-- ocr-supplement-end: x -->\n"
    assert ml.markup_free(md + supplement, layout) == text + supplement     # an appended OCR supplement is kept
    assert ml.markup_free(md.replace("body", "BODY"), layout) is None       # a changed file
    assert ml.markup_free("## Other\n\ntext", layout) is None                 # a record of another file
    assert ml.markup_free(md, None) is None and ml.markup_free(md, {"status": "labelled"}) is None
    broken = json.loads(json.dumps(layout))
    broken["strip"]["edits"][0][2] = "\n\n\n"
    assert ml.markup_free(md, broken) is None                                 # the plain text's hash is checked
    assert mlrender.strip_record(md, edits, "something else") is None


# ---- the extractor's record ----------------------------------------------------------------------------------------
from test_markdown_layout import _document_pdf, _extract, _legacy_pages  # noqa: E402


def test_extractor_strip_record_gives_the_plain_markdown(tmp_path, monkeypatch):
    pdf = _document_pdf(tmp_path / "doc.pdf")
    md, sidecar = _extract(pdf, tmp_path / "layout", monkeypatch)
    plain_md, plain_sidecar = _extract(pdf, tmp_path / "plain", monkeypatch, "plain")
    meta = sidecar["markdown_layout"]
    assert md != plain_md and meta["strip"]["schema"] == "glossapi.markdown_layout.strip.v1"
    assert ml.markup_free(md, meta) == plain_md == "\n\n".join(p for p in _legacy_pages(pdf) if p)
    assert "markdown_layout" not in plain_sidecar and plain_sidecar["decode_status"]["pages"] == ["complete"] * 4


def test_unavailable_labels_record_an_empty_strip(tmp_path, monkeypatch):
    import types
    pdf = _document_pdf(tmp_path / "doc.pdf", pages=2)
    stale = types.ModuleType("glossapi_rs_pdfrepair")
    stale.__glossapi_native_api__ = "GLOSSAPI_NATIVE_API:glossapi_rs_pdfrepair:2;"   # a K5-A2a-era build
    monkeypatch.setitem(sys.modules, "glossapi_rs_pdfrepair", stale)
    md, sidecar = _extract(pdf, tmp_path / "s", monkeypatch)
    meta = sidecar["markdown_layout"]
    assert meta["status"] == "unavailable" and meta["strip"]["edits"] == [] and ml.markup_free(md, meta) == md
    # the decode check needs the same level: every page unknown, with the reason
    assert sidecar["decode_status"]["pages"] == ["unknown", "unknown"] and "stale" in sidecar["decode_status"]["reason"]


# ---- cleaner metrics: default mode == plain mode -------------------------------------------------------------------
def _greek_document(path, pages=3):
    """A Greek document with a title, headings, lists, captions, a table and page numbers: lots of markup."""
    from pdf_fixtures import FixturePdf
    fx = FixturePdf()
    font = fx.load_truetype(_greek_font())
    para = ("Η ελληνική γλώσσα είναι μία από τις αρχαιότερες γλώσσες της Ευρώπης και γράφεται\n"
            "με το ελληνικό αλφάβητο από τον όγδοο αιώνα πριν από την εποχή μας και έως σήμερα.\n"
            "Το τέλος της παραγράφου.")
    for k in range(pages):
        page = fx.new_page(595, 842)
        if k == 0:
            fx.insert_text(page, (72, 70), "Μελέτη της διάταξης", fontsize=24, font=font)
        fx.insert_text(page, (72, 130), f"{k + 1}. Ενότητα αριθμός {k + 1}", fontsize=16, font=font)
        fx.insert_text(page, (72, 170), para, fontsize=10, font=font)
        fx.insert_text(page, (72, 230), para, fontsize=10, font=font)
        fx.insert_text(page, (72, 290), "- πρώτο σημείο της λίστας\n- δεύτερο σημείο της λίστας", fontsize=10, font=font)
        fx.insert_text(page, (150, 340), f"Πίνακας {k + 1}: Αποτελέσματα των μετρήσεων", fontsize=9, font=font)
        fx.insert_text(page, (100, 360), "Άλφα    10    20    30\nΒήτα    11    21    31", fontsize=10, font=font)
        fx.insert_text(page, (290, 800), str(k + 1), fontsize=10, font=font)
    return fx.save(path)


def _greek_font():
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(candidate).is_file():
            return candidate
    pytest.skip("no TrueType font with Greek glyphs on this machine")


def _corpus_metrics(tmp_path, name, mode, monkeypatch, *, profile=None, run=False, drop_strip=False):
    import os
    root = tmp_path / name
    (root / "in").mkdir(parents=True)
    for index in range(2):
        _greek_document(root / "in" / f"doc{index}.pdf", pages=2 + index)
    monkeypatch.setenv(ml.ENVIRONMENT, mode)
    if profile:
        monkeypatch.setenv("GLOSSAPI_STAGE_PROFILE", profile)
    else:
        monkeypatch.delenv("GLOSSAPI_STAGE_PROFILE", raising=False)
    from glossapi import Corpus
    if run:
        corpus = Corpus(root / "out", root / "out")
        corpus.run([str(p) for p in sorted((root / "in").glob("*.pdf"))], stages=["extract", "clean"],
                   settings={"debug": "off"})
    else:
        corpus = Corpus(root / "in", root / "out")
        corpus.extract(input_format="pdf", phase1_backend="safe", accel_type="CPU")
        if drop_strip:   # control: the sidecar without its strip record (K5-A2a's metadata)
            for sidecar in (root / "out/sidecars/extract").glob("*.json"):
                payload = json.loads(sidecar.read_text())
                payload["markdown_layout"].pop("strip")
                sidecar.write_text(json.dumps(payload, ensure_ascii=False))
        corpus.clean(num_threads=1)
    frame = pd.read_parquet(root / "out/download_results/download_results.parquet").set_index("filename").sort_index()
    os.environ.pop("GLOSSAPI_STAGE_PROFILE", None)
    return frame, root


def _columns(frame):
    return frame[[column for column in (*METRICS, *DECISIONS) if column in frame.columns]]


@pytest.mark.parametrize("profile", [None, "legacy"])
def test_clean_metrics_and_decisions_equal_the_plain_mode(tmp_path, monkeypatch, profile):
    plain, _ = _corpus_metrics(tmp_path, "plain", "plain", monkeypatch, profile=profile)
    layout, root = _corpus_metrics(tmp_path, "layout", "layout_v2", monkeypatch, profile=profile)
    assert (root / "out/markdown/doc0.md").read_text().startswith("# ")       # the Markdown carries markup
    pd.testing.assert_frame_equal(_columns(layout), _columns(plain))
    assert plain["percentage_greek"].min() > 50
    # the cleaned Markdown (the product) keeps its markup
    assert "<!-- caption -->" in (root / "out/clean_markdown/doc0.md").read_text()
    assert not list((root / "out/.glossapi/staging").glob("clean-metric-surface-*"))   # scratch copies removed


def test_control_metrics_move_without_the_strip_record(tmp_path, monkeypatch):
    plain, _ = _corpus_metrics(tmp_path, "plain", "plain", monkeypatch)
    control, _ = _corpus_metrics(tmp_path, "control", "layout_v2", monkeypatch, drop_strip=True)
    differ = [c for c in ("latin_percentage", "percentage_greek", "char_count_no_comments")
              if not control[c].equals(plain[c])]
    assert differ, "markup must move the metrics when the strip record is absent (the test would be blind otherwise)"


def test_clean_contract_follows_the_markdown_mode(monkeypatch):
    from glossapi.stages.clean import CleanStage
    assert CleanStage().contract_version == 2
    monkeypatch.setenv(ml.ENVIRONMENT, "plain")
    assert CleanStage().contract_version == 1     # plain mode: the clean identity of K5-A2a and before


def test_run_clean_binds_the_layout_input_and_equals_the_plain_mode(tmp_path, monkeypatch):
    import sqlite3
    plain, plain_root = _corpus_metrics(tmp_path, "plain", "plain", monkeypatch, run=True)
    with sqlite3.connect(plain_root / "out/corpus.sqlite3") as connection:
        assert ("layout_metadata",) not in connection.execute("SELECT DISTINCT input_role FROM task_dependencies").fetchall()
    layout, root = _corpus_metrics(tmp_path, "layout", "layout_v2", monkeypatch, run=True)
    pd.testing.assert_frame_equal(_columns(layout), _columns(plain))
    with sqlite3.connect(root / "out/corpus.sqlite3") as connection:
        rows = connection.execute(
            "SELECT t.operation, COUNT(*) FROM attempts a JOIN tasks t ON t.task_id=a.task_id "
            "WHERE t.operation='clean.initial' GROUP BY 1").fetchall()
        roles = connection.execute("SELECT DISTINCT input_role FROM task_dependencies").fetchall()
    assert rows == [("clean.initial", 2)]     # the publication replayed the DAG's attempts (same two inputs)
    assert ("layout_metadata",) in roles


def _clean_request(tmp_path, name, text, layout=None, operation="clean.score_after_ocr"):
    import hashlib
    import time
    import uuid
    from glossapi.stages.contracts import ArtifactReference, StageRequest
    work = tmp_path / name
    (work / "src").mkdir(parents=True)
    source = work / "src" / "doc.md"
    source.write_text(text, encoding="utf-8")
    inputs = [ArtifactReference("f1", "selected_text", source, hashlib.sha256(source.read_bytes()).hexdigest(),
                                source.stat().st_size)]
    if layout is not None:
        sidecar = work / "src" / "doc.json"
        sidecar.write_text(json.dumps({"markdown_layout": layout}))
        inputs.append(ArtifactReference("f2", "layout_metadata", sidecar,
                                        hashlib.sha256(sidecar.read_bytes()).hexdigest(), sidecar.stat().st_size))
    return StageRequest(*(str(uuid.uuid4()) for _ in range(3)), "d", "v", operation, tuple(inputs), {}, {}, "cpu",
                        work / "attempt", time.monotonic_ns() + 600 * 10**9)


@pytest.mark.parametrize("operation", ["clean.score_after_ocr", "clean.initial"])
def test_clean_stage_after_ocr_measures_native_plain_text_plus_supplement(tmp_path, operation):
    from glossapi.stages.clean import CleanStage
    from glossapi.stages.execution import ReceiptCollector
    text = ("Ενότητα πρώτη\nΗ ελληνική γλώσσα είναι αρχαία.\nΚαι συνεχίζει εδώ με περισσότερο κείμενο.\n"
            "Πίνακας 1: Τιμές\nΣημείωση στο κάτω μέρος")
    blocks = [(0, 13, "section_header", 1), (14, 83, "paragraph", None), (84, 100, "caption", None),
              (101, len(text), "footnote", None)]
    md, _, edits = mlrender.render_page_with_edits(text, blocks)
    layout = {"rules": mlrender.RULES, "strip": mlrender.strip_record(md, edits, text)}
    supplement = ("\n\n<!-- ocr-pages: 2 -->\n<!-- ocr-supplement-begin: pages=2 run_id=x -->\n<!-- page:2 -->\n"
                  "Κείμενο από την αναγνώριση.\n<!-- ocr-supplement-end: pages=2 run_id=x -->\n<!-- /ocr-pages -->\n")
    plain = CleanStage().run_document(_clean_request(tmp_path, "plain", text + supplement, operation=operation),
                                      reporter=ReceiptCollector())
    labelled = CleanStage().run_document(_clean_request(tmp_path, "layout", md + supplement, layout, operation),
                                         reporter=ReceiptCollector())
    assert plain.status == labelled.status == "ok", (plain.error, labelled.error)
    a, b = plain.decisions[0]["measurements"], labelled.decisions[0]["measurements"]
    assert a["native_report"] == b["native_report"] and a["noise_score"] == b["noise_score"]
    assert a["noise_surface_sha256"] == b["noise_surface_sha256"]
    assert b["metric_surface"]["basis"] == "markup_free_text"
    # a record that does not describe the text is not applied (metrics on the text, with the reason)
    other = {"rules": mlrender.RULES, "strip": mlrender.strip_record("x", [], "x")}
    unmatched = CleanStage().run_document(_clean_request(tmp_path, "other", md + supplement, other, operation),
                                          reporter=ReceiptCollector())
    assert unmatched.decisions[0]["measurements"]["metric_surface"]["basis"] == "text"
    if operation == "clean.initial":   # the cleaned product keeps the markup
        assert (tmp_path / "layout/attempt/clean_markdown/doc.md").read_text().startswith("## Ενότητα")
        assert not (tmp_path / "layout/attempt/layout_metrics").exists()


# ---- P4a: all-GLYPH text -> OCR -----------------------------------------------------------------------------------
def test_glyph_only_rule_uses_the_cleaners_matchers():
    from glossapi.corpus.phase_clean import glyph_only, GLYPH_ONLY_MIN_SHARE, GLYPH_ONLY_MIN_MATCHES
    import glossapi_rs_cleaner
    assert (GLYPH_ONLY_MIN_MATCHES, GLYPH_ONLY_MIN_SHARE) == (10, 0.9)
    tokens = " ".join(f"GLYPH&lt;{n}&gt;" for n in range(1, 19))
    assert glyph_only("## GLYPH ARTEFACT\n\n" + tokens)
    assert glyph_only(" ".join(f"/uni{0x3b1 + n:04X}" for n in range(12)))
    prose = "Η ελληνική γλώσσα είναι μία από τις αρχαιότερες γλώσσες της Ευρώπης. " * 30
    assert not glyph_only(prose + tokens)                       # readable text with some residue: not glyph-only
    # below the match floor: 4 markers are 8 matches as the cleaner counts them (GLYPH<n> is a Rule B and a bare match)
    assert not glyph_only(" ".join(f"GLYPH<{n}>" for n in range(4)))
    assert not glyph_only("")
    matches, residue, total = glossapi_rs_cleaner.glyph_residue_counts("GLYPH<1> x")
    assert (residue, total) == (8, 9) and matches >= 1


def test_glyph_only_document_is_routed_to_ocr_and_ordinary_text_is_not(tmp_path):
    from glossapi import Corpus
    out = tmp_path / "out"
    (out / "markdown").mkdir(parents=True)
    tokens = "\n".join(" ".join(f"GLYPH&lt;{6 * line + n}&gt;" for n in range(1, 7)) for line in range(3))
    (out / "markdown" / "glyphs.md").write_text("## GLYPH ARTEFACT\n\n" + tokens + "\n", encoding="utf-8")
    (out / "markdown" / "prose.md").write_text("Η ελληνική γλώσσα είναι αρχαία και ζωντανή. " * 40 + "GLYPH<1>",
                                               encoding="utf-8")
    Corpus(tmp_path / "in", out).clean(num_threads=1)
    frame = pd.read_parquet(out / "download_results/download_results.parquet").set_index("filename")
    assert frame.loc["glyphs.pdf", "filter"].split(";") == ["glyph_only"] and frame.loc["glyphs.pdf", "needs_ocr"]
    assert frame.loc["prose.pdf", "filter"] == "ok" and not frame.loc["prose.pdf", "needs_ocr"]
    assert "_glyph_only" not in frame.columns


# ---- P4b: decode_status -------------------------------------------------------------------------------------------
def _corrupt_streams(data):
    """``k5_prep/pageset/make_derived.py`` (the K5P-093 derivation): the middle third of every FlateDecode stream
    overwritten by 0x00."""
    out = bytearray(data)
    for m in re.finditer(rb"/FlateDecode.*?stream\r?\n", data, re.S):
        start = m.end()
        end = data.find(b"endstream", start)
        if end <= start:
            continue
        a, b = start + (end - start) // 3, start + 2 * (end - start) // 3
        out[a:b] = b"\x00" * (b - a)
    return bytes(out)


def _long_page(path):
    """One page of 40 rows of varied words (a stream long enough that its first third holds whole text operators)."""
    import random
    from pdf_fixtures import FixturePdf
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()
    pick = random.Random(20261104)
    fx = FixturePdf()
    page = fx.new_page(595, 842)
    for row in range(40):
        fx.insert_text(page, (40, 40 + 19 * row), f"{row:02d} " + " ".join(pick.choice(words) for _ in range(9)),
                       fontsize=10)
    return fx.save(path)


def _short_page(path):
    from pdf_fixtures import FixturePdf
    fx = FixturePdf()
    fx.insert_text(fx.new_page(595, 842), (72, 72), "Short.", fontsize=10)
    return fx.save(path)


def test_decode_status_partial_complete_unknown(tmp_path, monkeypatch):
    from glossapi.pdf_repair.decode_status import page_decode_status
    from glossapi.stages.receipts import capture_backend_observations
    good = _long_page(tmp_path / "good.pdf")
    data = good.read_bytes()
    assert b"/FlateDecode" in data
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(_corrupt_streams(data))
    with capture_backend_observations() as observations:
        md, sidecar = _extract(corrupt, tmp_path / "c", monkeypatch)
    assert sidecar["decode_status"]["pages"] == ["partial"] and sidecar["extract_status"] == "ok"
    assert "FlateDecode" in sidecar["decode_status"]["reasons"]["1"]
    assert 0 < len(md) < len("\n".join(_legacy_pages(good)))     # the readable prefix is kept (policy P4b)
    coverage = next(o["details"]["page_coverage"] for o in observations if "page_coverage" in o.get("details", {}))
    assert coverage["decode_status"] == ["partial"]          # the ledger's page details
    short = _short_page(tmp_path / "short.pdf")
    assert page_decode_status(short, 1)["pages"] == ["complete"]
    _, sidecar = _extract(short, tmp_path / "s", monkeypatch, "plain")
    assert sidecar["decode_status"] == {"schema": "glossapi.decode_status.v1",
                                        "check": "glossapi_rs_pdfrepair.page_decode_status_json",
                                        "pages": ["complete"], "reasons": {}}
    # the check cannot run: the native parser refuses the document, its page count disagrees, or the module is stale
    html = tmp_path / "x.pdf"
    html.write_bytes(b"<html>503</html>")
    status = page_decode_status(html, 2)
    assert status["pages"] == ["unknown", "unknown"] and "cannot read" in status["reason"]
    assert page_decode_status(short, 3)["pages"] == ["unknown"] * 3
    monkeypatch.setattr(mlr, "_native_state", (None, "stale native module glossapi_rs_pdfrepair"))
    assert page_decode_status(short, 1) == {"schema": "glossapi.decode_status.v1",
                                            "check": "glossapi_rs_pdfrepair.page_decode_status_json",
                                            "pages": ["unknown"], "reasons": {},
                                            "reason": "stale native module glossapi_rs_pdfrepair"}


def test_k5p093_document_is_partial_when_present():
    """The K5P-093 document itself (experiment input, not shipped): skipped where it is absent."""
    import os
    from glossapi.pdf_repair.decode_status import page_decode_status
    path = Path(os.environ.get("GLOSSAPI_K5P093_PDF", "/nonexistent"))
    if not path.is_file():
        pytest.skip("set GLOSSAPI_K5P093_PDF to the K5P-093 page (k5_prep/pageset/derived/F2_corrupt_WWZ_685_p4.pdf)")
    assert page_decode_status(path, 1)["pages"] == ["partial"]


# ---- P4d: Corpus.ocr follows run for a flagged document without routed pages ---------------------------------------
def test_math_only_still_excludes_every_flagged_document(tmp_path, monkeypatch):
    """P4d changes the OCR selection only: math_only keeps excluding cleaner-flagged documents, routed or not."""
    from test_k4g_b2_legacy import _quality_report
    from glossapi.corpus.ocr.targets import build_ocr_selection
    from glossapi.corpus.phase_clean import _attach_page_quality_routes
    from glossapi import Corpus
    metrics = tmp_path / "json" / "metrics"
    metrics.mkdir(parents=True)
    rows = []
    for name, flagged, routed in (("a.pdf", True, set()), ("b.pdf", True, {1}), ("c.pdf", False, set())):
        (metrics / f"{Path(name).stem}.page_quality.json").write_text(json.dumps(_quality_report(name, 2, routed)))
        rows.append(dict(filename=name, needs_ocr=flagged, filter="ok", page_count=2))
    target = tmp_path / "download_results" / "download_results.parquet"
    target.parent.mkdir(parents=True)
    _attach_page_quality_routes(pd.DataFrame(rows), metrics).to_parquet(target, index=False)
    corpus = Corpus(tmp_path, tmp_path)
    assert build_ocr_selection(corpus, mode="math_only", reprocess_completed=False).bad_files == ["a.pdf", "b.pdf"]
    assert build_ocr_selection(corpus, mode="ocr_bad", reprocess_completed=False).bad_files == ["b.pdf"]
    # without a page-quality report the document keeps document-level routing
    (metrics / "a.page_quality.json").unlink()
    _attach_page_quality_routes(pd.DataFrame(rows), metrics).to_parquet(target, index=False)
    corpus = Corpus(tmp_path, tmp_path)
    assert build_ocr_selection(corpus, mode="ocr_bad", reprocess_completed=False).bad_files == ["a.pdf", "b.pdf"]


def test_glyph_only_flagged_document_is_sent_to_ocr_whole(tmp_path, monkeypatch):
    """P4a with P4d: page quality v4 does not route non-Greek symbol pages, so an all-GLYPH document whose report routes
    no page is still dispatched, without page ranges: the OCR stage's glyph check then selects every page."""
    from test_k4g_b2_legacy import _quality_report
    from test_t1_4_pipeline import make_pdf
    from glossapi.corpus.ocr import dispatch as runner
    from glossapi.corpus.phase_clean import _attach_page_quality_routes
    from glossapi import Corpus
    metrics = tmp_path / "json" / "metrics"
    metrics.mkdir(parents=True)
    rows = []
    for name, flt, routed in (("glyphs.pdf", "glyph_only", {1}), ("noise.pdf", "mojibake>0.1", set())):
        make_pdf(tmp_path / name, 3)
        (metrics / f"{Path(name).stem}.page_quality.json").write_text(json.dumps(_quality_report(name, 3, routed)))
        rows.append(dict(filename=name, needs_ocr=True, filter=flt, page_count=3))
    target = tmp_path / "download_results" / "download_results.parquet"
    target.parent.mkdir(parents=True)
    _attach_page_quality_routes(pd.DataFrame(rows), metrics).to_parquet(target, index=False)
    corpus = Corpus(tmp_path, tmp_path)
    monkeypatch.setattr(corpus, "clean", lambda **kw: None)
    calls = []
    monkeypatch.setattr(runner, "run_plugin_ocr", lambda self, files, **kw: calls.append((list(files), kw)))
    corpus.ocr(mode="ocr_bad", math_enhance=False)
    files, kwargs = calls[0]
    assert files == ["glyphs.pdf"] and not kwargs.get("page_ranges")   # the routed page 1 is not a restriction


def test_ocr_stage_glyph_check_reads_the_markup_free_native_text(tmp_path):
    import hashlib
    from glossapi.stages.contracts import ArtifactReference
    from glossapi.stages.ocr import _native_text_is_glyph_only

    def ref(kind, path):
        return ArtifactReference(kind, kind, path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
    tokens = "\n".join(" ".join(f"GLYPH<{6 * row + n}>" for n in range(6)) for row in range(3))
    text = "GLYPH ARTEFACT\n" + tokens
    md, _, edits = mlrender.render_page_with_edits(text, [(0, 14, "title", None), (15, len(text), "paragraph", None)])
    native = tmp_path / "doc.md"
    native.write_text(md)
    sidecar = tmp_path / "doc.json"
    sidecar.write_text(json.dumps({"markdown_layout": {"strip": mlrender.strip_record(md, edits, text)}}))
    assert _native_text_is_glyph_only([ref("native_markdown", native), ref("extract_sidecar", sidecar)])
    prose = tmp_path / "prose.md"
    prose.write_text("Η ελληνική γλώσσα είναι αρχαία. " * 20)
    assert not _native_text_is_glyph_only([ref("native_markdown", prose)])
    assert not _native_text_is_glyph_only([ref("source_payload", prose)])   # no native text: no glyph route


SUPPLEMENT = ("\n\n<!-- ocr-pages: 2 -->\n<!-- ocr-supplement-begin: pages=2 run_id=x -->\n<!-- page:2 -->\n"
              "Κείμενο από την αναγνώριση χαρακτήρων.\n<!-- ocr-supplement-end: pages=2 run_id=x -->\n<!-- /ocr-pages -->\n")


def _refresh_after_ocr(tmp_path, name, mode, monkeypatch, profile):
    """Extract, then the cleaner refresh that follows OCR (clean_ocr + score-only clean, as refresh_cleaner_after_ocr
    runs it) on a fresh corpus whose Markdown carries an appended OCR supplement."""
    from glossapi import Corpus
    monkeypatch.setenv(ml.ENVIRONMENT, mode)
    if profile:
        monkeypatch.setenv("GLOSSAPI_STAGE_PROFILE", profile)
    else:
        monkeypatch.delenv("GLOSSAPI_STAGE_PROFILE", raising=False)
    source = tmp_path / f"{name}-extract"
    (source / "in").mkdir(parents=True)
    _greek_document(source / "in" / "doc0.pdf", pages=14)
    _greek_document(source / "in" / "doc1.pdf", pages=3)
    Corpus(source / "in", source / "out").extract(input_format="pdf", phase1_backend="safe", accel_type="CPU")
    root = tmp_path / name
    (root / "out").mkdir(parents=True)
    for sub in ("markdown", "sidecars/extract"):
        shutil.copytree(source / "out" / sub, root / "out" / sub)
    md = root / "out/markdown/doc0.md"
    md.write_text(md.read_text() + SUPPLEMENT, encoding="utf-8")
    corpus = Corpus(root / "in", root / "out")
    corpus.clean(num_threads=1)
    corpus.clean_ocr(input_dir=root / "out/markdown", drop_bad=False)
    strips = dict(getattr(corpus, "_clean_strips", {}))
    corpus.clean(input_dir=corpus.cleaned_markdown_dir, drop_bad=False, write_cleaned_files=False)
    frame = pd.read_parquet(root / "out/download_results/download_results.parquet").set_index("filename").sort_index()
    return frame, root, strips


OCR_METRICS = ("ocr_noise_suspect", "ocr_noise_flags", "ocr_repeat_phrase_run_max", "ocr_repeat_line_run_max",
               "ocr_repeat_suspicious_line_count", "ocr_repeat_suspicious_line_ratio")


def _markup_sensitive_ocr_cleaner(monkeypatch):
    """The real OCR cleaner plus one rule that reads layout markup (it drops the text line after a caption comment), so
    its output on the labelled Markdown differs from its output on the plain text by more than the markup - as the
    real cleaner's repetition rules do on some documents (seen on two of six page-set documents)."""
    import glossapi.corpus.phase_clean as pc
    from glossapi.stages.edits import _ocr_edit_sink
    real = pc._process_combined_ocr_document

    def cleaner(source_path, *, clean_output_path=None, **kwargs):
        details = real(source_path, clean_output_path=clean_output_path, **kwargs)
        if clean_output_path is not None:
            out = Path(clean_output_path)
            text = out.read_text(encoding="utf-8")
            new = re.sub(r"(?m)^<!-- caption -->\n[^\n]*", "<!-- caption -->", text)
            if new != text:
                out.write_text(new, encoding="utf-8")
                sink = _ocr_edit_sink.get()
                if sink is not None:
                    sink.recorder.transform("test_markup_sensitive_rule", new)
        return details
    monkeypatch.setattr(pc, "_process_combined_ocr_document", cleaner)


@pytest.mark.parametrize("profile", [None, "legacy"])
def test_metrics_after_the_ocr_refresh_equal_the_plain_mode(tmp_path, monkeypatch, profile):
    from glossapi import Corpus
    _markup_sensitive_ocr_cleaner(monkeypatch)
    plain, _, _ = _refresh_after_ocr(tmp_path, "plain", "plain", monkeypatch, profile)
    layout, root, strips = _refresh_after_ocr(tmp_path, "layout", "layout_v2", monkeypatch, profile)
    columns = [c for c in (*METRICS, *DECISIONS, *OCR_METRICS) if c in plain.columns]
    pd.testing.assert_frame_equal(layout[columns], plain[columns])
    # the OCR cleaner's strip records described its cleaned Markdown (which keeps the markup) for both documents
    assert len(strips) == 2 and (root / "out/clean_markdown/doc0.md").read_text().startswith("# ")
    # control: without the OCR cleaner's strip records the score-only pass measures other text
    real = Corpus.clean_ocr

    def without_handoff(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        self._clean_strips = {}
        return result
    monkeypatch.setattr(Corpus, "clean_ocr", without_handoff)
    control, _, _ = _refresh_after_ocr(tmp_path, "control", "layout_v2", monkeypatch, profile)
    assert [c for c in ("char_count_no_comments", "percentage_greek") if not control[c].equals(plain[c])]
