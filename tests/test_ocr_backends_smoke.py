from pathlib import Path

import pandas as pd


def _mk_corpus(tmp_path: Path):
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_cross_backend_smoke_with_stubs(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    # Two PDFs: one needs OCR, one does not (for math-only later)
    (corpus.input_dir / "needs.pdf").write_bytes(b"%PDF-1.4\n%stub\n")
    (corpus.input_dir / "clean.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

    # Seed metadata
    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"filename": "needs.pdf", corpus.url_column: "", "needs_ocr": True, "ocr_success": False},
        {"filename": "clean.pdf", corpus.url_column: "", "needs_ocr": False, "ocr_success": True},
    ])
    parquet_path = dl_dir / "download_results.parquet"
    df.to_parquet(parquet_path, index=False)

    # DeepSeek stub for OCR
    import glossapi.ocr.deepseek_runner as runner

    def fake_run_for_files(self_ref, files, **kwargs):
        for f in files:
            stem = Path(f).stem
            (corpus.output_dir / "markdown").mkdir(parents=True, exist_ok=True)
            (corpus.output_dir / "json" / "metrics").mkdir(parents=True, exist_ok=True)
            (corpus.output_dir / "markdown" / f"{stem}.md").write_text("ds md\n", encoding="utf-8")
            (corpus.output_dir / "json" / "metrics" / f"{stem}.metrics.json").write_text("{\n \"page_count\": 1\n}\n", encoding="utf-8")
        return {"needs": {"page_count": 1}}

    monkeypatch.setattr(runner, "run_for_files", fake_run_for_files)

    # Run DeepSeek OCR for bad files
    corpus.ocr(backend="deepseek", fix_bad=True, math_enhance=True, mode="ocr_bad_then_math")

    # RapidOCR math-only pass: ensure JSON for clean.pdf and run math
    json_dir = corpus.output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / "clean.docling.json").write_text("{}", encoding="utf-8")

    captured = {}

    def fake_enrich(files=None, **kwargs):
        captured["files"] = list(files or [])
        return None

    monkeypatch.setattr(corpus, "formula_enrich_from_json", fake_enrich)

    corpus.ocr(backend="rapidocr", fix_bad=False, math_enhance=True, mode="math_only")

    # Verify
    updated = pd.read_parquet(parquet_path).set_index("filename")
    assert bool(updated.loc["needs.pdf", "ocr_success"]) is True
    assert captured.get("files") == ["clean"], "Math-only should run for non-OCR stem only"

