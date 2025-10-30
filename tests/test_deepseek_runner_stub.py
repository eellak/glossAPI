from pathlib import Path

import pandas as pd


def _mk_corpus(tmp_path: Path):
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_deepseek_backend_stub_runs_and_updates_parquet(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    # Seed a minimal metadata parquet with one bad file
    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    df = pd.DataFrame(
        [{"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}]
    )
    parquet_path = dl_dir / "download_results.parquet"
    df.to_parquet(parquet_path, index=False)

    # Create an empty placeholder file for the PDF
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%stub\n")

    # Monkeypatch the runner internal to avoid heavy imports
    from glossapi.ocr.deepseek import runner

    def fake_run_one(pdf_path, md_out, metrics_out, cfg):
        md_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text("deepseek stub output\n", encoding="utf-8")
        metrics_out.write_text("{\n  \"page_count\": 1\n}\n", encoding="utf-8")
        return {"page_count": 1}

    monkeypatch.setattr(runner, "_run_one_pdf", fake_run_one)

    # Run OCR via dispatcher
    corpus.ocr(backend="deepseek", fix_bad=True, math_enhance=False)

    # Artifacts exist
    stem = "doc"
    md = corpus.output_dir / "markdown" / f"{stem}.md"
    metrics = corpus.output_dir / "json" / "metrics" / f"{stem}.metrics.json"
    assert md.exists(), "Markdown output should be created by deepseek stub"
    assert metrics.exists(), "Metrics JSON should be created by deepseek stub"

    # Parquet updated
    updated = pd.read_parquet(parquet_path).set_index("filename")
    row = updated.loc[fname]
    assert bool(row["ocr_success"]) is True
    assert bool(row["needs_ocr"]) is False
    # extraction_mode is optional; if present assert value
    if "extraction_mode" in updated.columns:
        assert updated.loc[fname, "extraction_mode"] == "deepseek"
