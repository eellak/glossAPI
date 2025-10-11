import os
from pathlib import Path

import onnxruntime as ort
import pandas as pd
import pytest
import torch

from glossapi import Corpus
from glossapi.corpus import _resolve_skiplist_path
from fpdf import FPDF


pytest.importorskip("docling")
pytest.importorskip("glossapi_rs_cleaner")


_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _write_pdf(path: Path, text: str | None) -> None:
    """Create a one-page PDF using DejaVuSans to preserve non-ASCII text."""

    pdf = FPDF()
    pdf.add_page()
    if _FONT_PATH.exists():
        pdf.add_font("DejaVuSans", "", str(_FONT_PATH))
        pdf.set_font("DejaVuSans", "", 16)
    else:
        pdf.set_font("Helvetica", "", 16)

    content = (text or "").strip()
    if not content:
        # Leave the page mostly blank but add a tiny marker so OCR sees an empty page.
        pdf.ln(10)
    else:
        pdf.multi_cell(0, 10, content)

    pdf.output(str(path))


def _assert_dir_contents(
    root: Path,
    allowed_suffixes: set[str],
    *,
    allowed_dirs: set[str] = frozenset(),
    allowed_exact: set[str] = frozenset(),
) -> None:
    assert root.exists(), f"Missing directory: {root}"
    for entry in root.iterdir():
        if entry.is_dir():
            if entry.name in allowed_dirs:
                continue
            pytest.fail(f"Unexpected subdirectory {entry} in {root}")
        else:
            name = entry.name
            if name in allowed_exact:
                continue
            if not any(name.endswith(suffix) for suffix in allowed_suffixes):
                pytest.fail(f"Unexpected file {entry} in {root}")


def test_pipeline_smoke_and_artifacts(tmp_path):
    assert torch.cuda.is_available(), "CUDA GPU expected for pipeline smoke test"
    providers = ort.get_available_providers()
    assert "CUDAExecutionProvider" in providers, f"CUDAExecutionProvider missing: {providers}"

    device_idx = 0
    if torch.cuda.device_count() > 1:
        device_idx = torch.cuda.current_device()

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    good_pdf = corpus_dir / "text.pdf"
    blank_pdf = corpus_dir / "blank.pdf"
    _write_pdf(good_pdf, "A simple line of text")
    _write_pdf(blank_pdf, None)

    corpus = Corpus(input_dir=corpus_dir, output_dir=corpus_dir)

    corpus.extract(
        input_format="pdf",
        accel_type="CUDA",
        num_threads=1,
        emit_formula_index=True,
        phase1_backend="docling",
        force_ocr=True,
        use_gpus="single",
        devices=[device_idx],
    )

    corpus.clean()

    parquet_path = corpus_dir / "download_results" / "download_results.parquet"
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    needs = df.set_index("filename").get("needs_ocr")
    assert bool(needs.get("blank.pdf")), "Blank PDF should be flagged for OCR"
    assert not bool(needs.get("text.pdf"))

    corpus.ocr(
        mode="ocr_bad",
        use_gpus="single",
        devices=[device_idx],
        math_enhance=False,
    )

    corpus.section()

    markdown_dir = corpus_dir / "markdown"
    clean_markdown_dir = corpus_dir / "clean_markdown"
    json_dir = corpus_dir / "json"
    metrics_dir = json_dir / "metrics"
    downloads_dir = corpus_dir / "downloads"
    sections_dir = corpus_dir / "sections"

    _assert_dir_contents(downloads_dir, {".pdf"})
    _assert_dir_contents(
        markdown_dir,
        {".md"},
        allowed_dirs={"problematic_files", "timeout_files"},
        allowed_exact={".processing_state.pkl"},
    )
    _assert_dir_contents(
        clean_markdown_dir,
        {".md"},
        allowed_dirs={"timeout_files", "problematic_files"},
        allowed_exact={".processing_state.pkl"},
    )
    if json_dir.exists():
        _assert_dir_contents(
            json_dir,
            {".docling.json", ".docling.json.zst", ".formula_index.jsonl"},
            allowed_dirs={"metrics"},
        )
        if metrics_dir.exists():
            _assert_dir_contents(metrics_dir, {".json"})
    _assert_dir_contents(corpus_dir / "download_results", {".parquet"})
    _assert_dir_contents(sections_dir, {".parquet"})

    sections_file = sections_dir / "sections_for_annotation.parquet"
    assert sections_file.exists()


def test_docling_math_pipeline_with_mixed_pdfs(tmp_path, monkeypatch):
    assert torch.cuda.is_available(), "CUDA GPU expected for docling pipeline test"
    providers = ort.get_available_providers()
    assert "CUDAExecutionProvider" in providers, f"CUDAExecutionProvider missing: {providers}"

    device_idx = 0
    if torch.cuda.device_count() > 1:
        device_idx = torch.cuda.current_device()

    monkeypatch.setenv("GLOSSAPI_GPU_BATCH_SIZE", "1")
    monkeypatch.setenv("GLOSSAPI_WORKER_LOG_VERBOSE", "0")

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    documents = {
        "blank": None,
        "greek_text": "Καλημέρα κόσμε",
        "math_latex": r"\[\int_0^\infty e^{-x^2} \, dx = \frac{\sqrt{\pi}}{2}\]",
        "greek_consonants": "".join(
            part
            for part in [
                "ββββ", "γγγγ", "δδδδ", "ζζζζ", "θθθθ", "κκκκ",
                "λλλλ", "μμμμ", "νννν", "ξξξξ", "ππππ", "ρρρρ",
                "σσσσ", "ςςςς", "ττττ", "φφφφ", "χχχχ", "ψψψψ",
            ]
        ),
    }

    for stem, content in documents.items():
        _write_pdf(corpus_dir / f"{stem}.pdf", content)

    corpus = Corpus(input_dir=corpus_dir, output_dir=corpus_dir)

    corpus.extract(
        input_format="pdf",
        accel_type="CUDA",
        num_threads=1,
        emit_formula_index=True,
        phase1_backend="docling",
        force_ocr=True,
        use_gpus="single",
        devices=[device_idx],
    )

    corpus.clean()

    parquet_path = corpus_dir / "download_results" / "download_results.parquet"
    results_after_clean = pd.read_parquet(parquet_path).set_index("filename")
    greek_row = results_after_clean.loc["greek_consonants.pdf"]
    assert greek_row["greek_badness_score"] > 60, "Expected high greek badness score"
    assert bool(greek_row["needs_ocr"]), "Greek consonant doc should require OCR rerun"
    assert "non_greek_text" in str(greek_row.get("filter", "")), "Filter should record non-Greek text"

    corpus.ocr(
        fix_bad=True,
        math_enhance=True,
        use_gpus="single",
        devices=[device_idx],
    )

    results_after_ocr = pd.read_parquet(parquet_path).set_index("filename")
    greek_after = results_after_ocr.loc["greek_consonants.pdf"]
    assert not bool(greek_after["needs_ocr"]), "Greek consonant doc should be resolved after OCR rerun"
    assert bool(greek_after.get("ocr_success", False)), "OCR rerun should mark greek consonant doc as success"

    json_dir = corpus_dir / "json"
    assert json_dir.exists(), "Docling JSON directory should exist after extraction"
    for stem in documents:
        if stem == "blank":
            # Blank documents may be routed to timeout/problematic, skip strict JSON check.
            continue
        json_plain = json_dir / f"{stem}.docling.json"
        json_zst = json_dir / f"{stem}.docling.json.zst"
        assert json_plain.exists() or json_zst.exists(), f"Missing Docling JSON for {stem}"

    math_sidecar = corpus_dir / "sidecars" / "math" / "math_latex.json"
    assert math_sidecar.exists(), "Expected math enrichment to produce math sidecar metrics"

    skiplist_path = _resolve_skiplist_path(corpus.output_dir, corpus.logger)
    if skiplist_path.exists():
        assert not skiplist_path.read_text(encoding="utf-8").strip(), "Fatal skip-list should remain empty"
