from pathlib import Path

import pandas as pd
import pytest


def _mk_corpus(tmp_path: Path):
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_deepseek_backend_rejects_stub_mode(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    df = pd.DataFrame(
        [{"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}]
    )
    parquet_path = dl_dir / "download_results.parquet"
    df.to_parquet(parquet_path, index=False)
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%real\n")

    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "1")

    with pytest.raises(RuntimeError, match="stub execution has been removed"):
        corpus.ocr(backend="deepseek", fix_bad=True, math_enhance=False)

    updated = pd.read_parquet(parquet_path).set_index("filename")
    assert bool(updated.loc[fname, "ocr_success"]) is False
    assert bool(updated.loc[fname, "needs_ocr"]) is True


def test_progress_artifacts_stay_out_of_canonical_markdown(tmp_path):
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _write_outputs, _write_progress

    output_dir = tmp_path / "output"
    _write_progress(
        output_dir=output_dir,
        stem="doc",
        page_outputs=["page one"],
        total_pages=5,
        completed_pages=1,
    )

    canonical_markdown = output_dir / "markdown" / "doc.md"
    progress_markdown = output_dir / "sidecars" / "ocr_progress" / "doc.partial.md"
    progress_json = output_dir / "json" / "metrics" / "doc.progress.json"

    assert not canonical_markdown.exists()
    assert progress_markdown.exists()
    assert progress_json.exists()

    _write_outputs(output_dir=output_dir, stem="doc", markdown="final", page_count=5)

    assert canonical_markdown.exists()
    assert canonical_markdown.read_text(encoding="utf-8") == "final\n"
    assert not progress_markdown.exists()
