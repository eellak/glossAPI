from pathlib import Path

import pandas as pd


def _mk_corpus(tmp_path: Path):
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_math_only_skips_docs_flagged_for_ocr(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    # Seed metadata: one needs OCR, one does not
    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {"filename": "needs.pdf", corpus.url_column: "", "needs_ocr": True, "ocr_success": False},
        {"filename": "clean.pdf", corpus.url_column: "", "needs_ocr": False, "ocr_success": True},
    ])
    df.to_parquet(dl_dir / "download_results.parquet", index=False)

    # Provide Docling JSON for both
    json_dir = corpus.output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / "needs.docling.json").write_text("{}", encoding="utf-8")
    (json_dir / "clean.docling.json").write_text("{}", encoding="utf-8")

    captured = {}

    def fake_enrich(files=None, **kwargs):
        captured["files"] = list(files or [])
        return None

    monkeypatch.setattr(corpus, "formula_enrich_from_json", fake_enrich)

    corpus.ocr(fix_bad=False, math_enhance=True, mode="math_only")

    assert captured.get("files") == ["clean"], "Math should run only for non-OCR stems"

