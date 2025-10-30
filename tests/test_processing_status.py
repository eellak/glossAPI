from pathlib import Path

import glossapi.corpus as corpus_mod
import glossapi.ocr.math as math_enrich_mod
from glossapi import Corpus
import pandas as pd


def test_ocr_skip_completed(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    corpus = Corpus(input_dir=input_dir, output_dir=output_dir)

    download_dir = output_dir / "download_results"
    download_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "filename": ["a.pdf", "b.pdf"],
            "needs_ocr": [True, True],
            "ocr_success": [True, False],
        }
    )
    df.to_parquet(download_dir / "download_results.parquet", index=False)

    processed = []

    def fake_extract(self, *args, **kwargs):
        filenames = list(kwargs.get("filenames", []) or [])
        processed.append(filenames)
        for fname in filenames:
            stem = Path(str(fname)).stem
            (self.markdown_dir / f"{stem}.md").write_text("md", encoding="utf-8")

    monkeypatch.setattr(Corpus, "extract", fake_extract, raising=False)
    monkeypatch.setattr(Corpus, "clean", lambda self, *a, **k: None, raising=False)

    corpus.ocr(fix_bad=True, math_enhance=False, reprocess_completed=False)
    assert processed == [["b.pdf"]]

    processed.clear()
    corpus.ocr(fix_bad=True, math_enhance=False, reprocess_completed=True)
    assert len(processed) == 1
    assert set(processed[0]) == {"a.pdf", "b.pdf"}


def test_math_skip_completed(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    corpus = Corpus(input_dir=input_dir, output_dir=output_dir)

    download_dir = output_dir / "download_results"
    download_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "filename": ["a.pdf", "b.pdf"],
            "needs_ocr": [False, False],
            "math_enriched": [True, False],
        }
    )
    df.to_parquet(download_dir / "download_results.parquet", index=False)

    json_dir = output_dir / "json"
    downloads_dir = output_dir / "downloads"
    json_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        (json_dir / f"{stem}.docling.json").write_text("{}", encoding="utf-8")
        (downloads_dir / f"{stem}.pdf").write_bytes(b"pdf")

    enriched = []

    def fake_enrich(json_path, pdf_path, out_md_path, out_map_path, **kwargs):
        enriched.append(out_md_path.stem)
        out_md_path.write_text("md", encoding="utf-8")
        out_map_path.write_text("", encoding="utf-8")
        return {"items": 1, "accepted": 1, "time_sec": 0.0}

    monkeypatch.setattr(corpus_mod, "enrich_from_docling_json", fake_enrich, raising=False)
    monkeypatch.setitem(math_enrich_mod.__dict__, "enrich_from_docling_json", fake_enrich)
    assert corpus_mod.enrich_from_docling_json is fake_enrich

    monkeypatch.setattr(Corpus, "clean", lambda self, *a, **k: None, raising=False)
    monkeypatch.setattr(Corpus, "extract", lambda self, *a, **k: None, raising=False)

    corpus.ocr(fix_bad=False, math_enhance=True, mode="math_only", reprocess_completed=False)
    assert enriched == ["b"]

    enriched.clear()
    corpus.ocr(fix_bad=False, math_enhance=True, mode="math_only", reprocess_completed=True)
    assert set(enriched) == {"a", "b"}

    updated = pd.read_parquet(download_dir / "download_results.parquet")
    assert bool(updated.loc[updated["filename"] == "b.pdf", "math_enriched"].iloc[0])
