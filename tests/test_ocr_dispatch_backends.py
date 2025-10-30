import os
from pathlib import Path

import pandas as pd
import pytest


def _mk_corpus(tmp_path: Path):
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_deepseek_backend_ignores_math_flags_and_runs_ocr_only(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    # Seed metadata with one bad file
    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    df = pd.DataFrame([
        {"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}
    ])
    df.to_parquet(dl_dir / "download_results.parquet", index=False)

    # Create stub pdf
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%stub\n")

    # Capture deepseek runner calls and assert math is not invoked
    from glossapi.ocr.deepseek import runner

    calls = {}

    def fake_run_for_files(self_ref, files, **kwargs):
        calls["files"] = list(files)
        return {"doc": {"page_count": 1}}

    monkeypatch.setattr(runner, "run_for_files", fake_run_for_files)

    # Also ensure formula_enrich_from_json would raise if called
    def fail_math(*args, **kwargs):
        raise AssertionError("Phase-2 math should not run for DeepSeek OCR targets")

    monkeypatch.setattr(corpus, "formula_enrich_from_json", fail_math)

    # Run with math flags present — should still just OCR the bad file
    corpus.ocr(backend="deepseek", fix_bad=True, math_enhance=True, mode="ocr_bad_then_math")

    assert calls.get("files") == [fname]


def test_rapidocr_backend_routes_to_extract_with_docling(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    # Seed minimal metadata parquet that flags a single file for OCR
    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"filename": "doc.pdf", corpus.url_column: "", "needs_ocr": True, "ocr_success": False}
    ])
    df.to_parquet(dl_dir / "download_results.parquet", index=False)

    captured = {}

    def fake_extract(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(corpus, "extract", fake_extract)

    corpus.ocr(backend="rapidocr", fix_bad=True, math_enhance=False, use_gpus="single", devices=[0])

    assert captured, "Expected extract() to be called for rapidocr backend"
    assert captured.get("force_ocr") is True
    assert captured.get("phase1_backend") == "docling"
    files = captured.get("filenames") or []
    assert files and files[0] == "doc.pdf"
