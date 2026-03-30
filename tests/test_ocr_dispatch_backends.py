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


def test_invalid_backend_is_rejected(tmp_path):
    corpus = _mk_corpus(tmp_path)
    with pytest.raises(ValueError, match="backend must be 'deepseek'"):
        corpus.ocr(backend="bogus", fix_bad=True, math_enhance=False)
