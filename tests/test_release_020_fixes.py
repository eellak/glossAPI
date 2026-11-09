"""FINAL-ASTRA (0.2.0 review) regressions: null-safe glyph_only selection (F02) and the default math API (F03)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from glossapi import Corpus
from glossapi.corpus.ocr.targets import build_ocr_selection
from glossapi.ledger.validation import validate_evidence

SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "lightweight_pdf_corpus" / "pdfs"
MISSING = [pytest.param(None, id="None"), pytest.param(float("nan"), id="NaN"), pytest.param(pd.NA, id="pd.NA")]


def _write_metadata(root: Path, rows: list[dict], *, filter_dtype: str | None = None) -> Corpus:
    frame = pd.DataFrame(rows)
    if filter_dtype is not None:
        frame["filter"] = frame["filter"].astype(filter_dtype)
    target = root / "download_results" / "download_results.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return Corpus(root, root)


# ---- F02: a missing filter value is unknown metadata, not a crash and not glyph_only ----------------------------
@pytest.mark.parametrize("missing", MISSING)
@pytest.mark.parametrize("dtype", [None, "str", "object"])
@pytest.mark.parametrize("mode", ["math_only", "ocr_bad"])
def test_f02_missing_filter_does_not_break_selection(tmp_path, missing, dtype, mode):
    corpus = _write_metadata(tmp_path, [dict(filename="alpha.pdf", needs_ocr=False, filter=missing)],
                             filter_dtype=dtype)
    assert build_ocr_selection(corpus, mode=mode, reprocess_completed=False).bad_files == []


@pytest.mark.parametrize("missing", MISSING)
def test_f02_mixed_metadata_keeps_glyph_only_and_ordinary_behaviour(tmp_path, missing):
    """Flagged documents whose page quality ran and routed no page: a glyph_only text layer still goes to OCR whole;
    an ordinary or missing filter is not OCRed (P4d). math_only keeps excluding every flagged document."""
    rows = [
        dict(filename="glyph.pdf", needs_ocr=True, filter="glyph_only", page_rule_profile="v6", ocr_pages=None),
        dict(filename="multi.pdf", needs_ocr=True, filter="ok;glyph_only", page_rule_profile="v6", ocr_pages=None),
        dict(filename="plain.pdf", needs_ocr=True, filter="ok", page_rule_profile="v6", ocr_pages=None),
        dict(filename="empty.pdf", needs_ocr=True, filter="", page_rule_profile="v6", ocr_pages=None),
        dict(filename="missing.pdf", needs_ocr=True, filter=missing, page_rule_profile="v6", ocr_pages=None),
        dict(filename="routed.pdf", needs_ocr=False, filter=missing, page_rule_profile="v6", ocr_pages="2"),
        dict(filename="clean.pdf", needs_ocr=False, filter="ok", page_rule_profile="v6", ocr_pages=None),
    ]
    corpus = _write_metadata(tmp_path, rows)
    ocr = build_ocr_selection(corpus, mode="ocr_bad", reprocess_completed=False)
    assert ocr.bad_files == ["glyph.pdf", "multi.pdf", "routed.pdf"]
    assert ocr.page_ranges == ["routed.pdf:2:2"]
    math = build_ocr_selection(corpus, mode="math_only", reprocess_completed=False)
    assert math.bad_files == ["glyph.pdf", "multi.pdf", "plain.pdf", "empty.pdf", "missing.pdf", "routed.pdf"]


def test_f02_math_only_after_lightweight_extraction(tmp_path):
    """The reviewed public path: extract the shipped PDFs (null filter metadata), then math-only selection."""
    source = tmp_path / "in"
    source.mkdir()
    for pdf in SAMPLES.glob("*.pdf"):
        shutil.copy2(pdf, source / pdf.name)
    corpus = Corpus(source, tmp_path / "out")
    corpus.extract(input_format="pdf", phase1_backend="safe", use_gpus="none")
    for mode in ("math_only", "ocr_bad"):
        assert build_ocr_selection(corpus, mode=mode, reprocess_completed=False).bad_files == []


# ---- F03: files=None is the auto-discovery selector, not an inventory --------------------------------------------
def test_f03_settings_snapshot_accepts_selector_none_inventories_stay_arrays():
    validate_evidence({"schema_version": 1, "kind": "settings", "files": None, "device": "cuda"})
    validate_evidence({"schema_version": 1, "kind": "settings", "files": ["a", "b"]})
    for kind in ("inputs", "outputs", "fingerprint"):
        with pytest.raises(ValueError, match="files must be an array"):
            validate_evidence({"schema_version": 1, "kind": kind, "files": None})
    with pytest.raises(ValueError, match="rows must be an array"):
        validate_evidence({"schema_version": 1, "rows": {"a": 1}})


@pytest.fixture
def math_calls(monkeypatch):
    """Replace the math backend (both the legacy and the adapter route) with a recorder; no Docling, no model."""
    import glossapi.ocr.math as math_package
    import glossapi.stages.compatibility as compatibility
    calls: list[str] = []
    stats = {"items": 0, "accepted": 0, "time_sec": 0.0}

    def legacy(json_path, pdf_path, *args, **kwargs):
        calls.append(Path(pdf_path).stem)
        return dict(stats)

    def adapter(corpus, source, document, **kwargs):
        calls.append(Path(source).stem)
        return dict(stats)

    # The package resolves names lazily (importing Docling); set the attribute without triggering that lookup.
    monkeypatch.setitem(vars(math_package), "enrich_from_docling_json", legacy)
    monkeypatch.setattr(compatibility, "math_compatibility_document", adapter)
    return calls


def _docling_inputs(root: Path, stems: list[str]) -> None:
    (root / "json").mkdir(parents=True, exist_ok=True)
    (root / "downloads").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (root / "json" / f"{stem}.docling.json").write_text("{}", encoding="utf-8")
        shutil.copy2(next(SAMPLES.glob("*.pdf")), root / "downloads" / f"{stem}.pdf")


def test_f03_omitted_files_on_a_fresh_corpus(tmp_path, math_calls):
    corpus = Corpus(tmp_path, tmp_path)
    assert corpus.formula_enrich_from_json() is None  # before the fix: ValueError: files must be an array
    assert math_calls == []


def test_f03_omitted_files_discovers_docling_json(tmp_path, math_calls):
    _docling_inputs(tmp_path, ["alpha", "beta"])
    Corpus(tmp_path, tmp_path).formula_enrich_from_json(device="cpu", batch_size=2)
    assert sorted(math_calls) == ["alpha", "beta"]


def test_f03_explicit_files_restrict_the_stems(tmp_path, math_calls):
    _docling_inputs(tmp_path, ["alpha", "beta"])
    Corpus(tmp_path, tmp_path).formula_enrich_from_json(files=["beta"], device="cpu")
    assert math_calls == ["beta"]
