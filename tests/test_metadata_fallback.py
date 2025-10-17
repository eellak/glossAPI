import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from glossapi._naming import canonical_stem
from glossapi.parquet_schema import ParquetSchema


def test_ensure_metadata_parquet_builds_from_artifacts(tmp_path):
    base_dir = tmp_path / "corpus"
    downloads_dir = base_dir / "downloads"
    markdown_dir = base_dir / "markdown"
    clean_dir = base_dir / "clean_markdown"
    json_dir = base_dir / "json"
    metrics_dir = json_dir / "metrics"
    triage_dir = base_dir / "sidecars" / "triage"
    math_sidecar_dir = base_dir / "sidecars" / "math"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    triage_dir.mkdir(parents=True, exist_ok=True)
    math_sidecar_dir.mkdir(parents=True, exist_ok=True)

    # Primary document with full artefacts
    (downloads_dir / "paper1.pdf").write_bytes(b"%PDF-1.4")
    (markdown_dir / "paper1.md").write_text("content", encoding="utf-8")
    (clean_dir / "paper1.md").write_text("clean content", encoding="utf-8")
    (json_dir / "paper1.latex_map.jsonl").write_text("", encoding="utf-8")

    metrics_payload = {
        "page_count": 5,
        "pages": [
            {"formula_count": 2},
            {"formula_count": 0},
            {"formula_count": 1},
        ],
    }
    (metrics_dir / "paper1.metrics.json").write_text(json.dumps(metrics_payload), encoding="utf-8")

    triage_payload = {
        "formula_total": 3,
        "formula_avg_pp": 1.0,
        "formula_p90_pp": 2,
        "pages_total": 3,
        "pages_with_formula": 2,
        "phase_recommended": "2A",
    }
    (triage_dir / "paper1.json").write_text(json.dumps(triage_payload), encoding="utf-8")

    math_payload = {"items": 4, "accepted": 3, "time_sec": 12.5}
    (math_sidecar_dir / "paper1.json").write_text(json.dumps(math_payload), encoding="utf-8")

    # Secondary document with only the downloaded PDF
    (downloads_dir / "paper2.pdf").write_bytes(b"%PDF-1.4")

    schema = ParquetSchema({"url_column": "url"})
    parquet_path = schema.ensure_metadata_parquet(base_dir)
    assert parquet_path is not None
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert set(df["filename"]) == {"paper1.pdf", "paper2.pdf"}

    row1 = df.loc[df["filename"] == "paper1.pdf"].iloc[0]
    assert bool(row1["download_success"])
    assert bool(row1["math_enriched"])
    assert "math" in (row1["processing_stage"] or "")
    assert row1["math_items"] == 4
    assert row1["math_accepted"] == 3
    assert row1["math_time_sec"] == 12.5
    assert row1["math_accept_rate"] == pytest.approx(0.75)
    assert row1["formula_total"] == 3
    assert row1["pages_total"] == 3
    assert row1["page_count"] == 5
    assert row1["phase_recommended"] == "2A"

    row2 = df.loc[df["filename"] == "paper2.pdf"].iloc[0]
    assert bool(row2["download_success"])
    assert not bool(row2["math_enriched"])
    assert "download" in (row2["processing_stage"] or "")

    # Subsequent ensure calls should be idempotent (existing parquet reused)
    parquet_again = schema.ensure_metadata_parquet(base_dir)
    assert parquet_again == parquet_path


def test_ensure_metadata_parquet_updates_existing(tmp_path):
    base_dir = tmp_path / "existing"
    download_dir = base_dir / "download_results"
    download_dir.mkdir(parents=True, exist_ok=True)

    existing = download_dir / "download_results_remaining.parquet"
    df = pd.DataFrame(
        {
            "filename": ["doc.pdf"],
            "url": ["http://example.com/doc.pdf"],
        }
    )
    df.to_parquet(existing, index=False)

    # create a minimal downloads tree so ensure can locate sidecar data if needed
    (base_dir / "downloads").mkdir(parents=True, exist_ok=True)

    schema = ParquetSchema({"url_column": "url"})
    result = schema.ensure_metadata_parquet(base_dir)

    assert result == existing
    assert not (download_dir / "download_results.parquet").exists()

    updated = pd.read_parquet(existing)
    assert "math_enriched" in updated.columns
    assert "ocr_success" in updated.columns


def _ensure_parquet_worker(base_dir: Path) -> None:
    schema = ParquetSchema({"url_column": "url"})
    schema.ensure_metadata_parquet(base_dir)


def test_ensure_metadata_parquet_concurrent(tmp_path):
    base_dir = tmp_path / "parallel"
    downloads_dir = base_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    for name in ("alpha.pdf", "beta.pdf"):
        (downloads_dir / name).write_bytes(b"%PDF-1.4")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_ensure_parquet_worker, base_dir) for _ in range(4)]
        for fut in futures:
            fut.result(timeout=10)

    parquet_path = base_dir / "download_results" / "download_results.parquet"
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert set(df["filename"]) == {"alpha.pdf", "beta.pdf"}


def test_write_metadata_parquet_normalizes_boolean(tmp_path):
    parquet_path = tmp_path / "bools.parquet"
    df = pd.DataFrame(
        {
            "filename": ["doc1.pdf", "doc2.pdf"],
            "needs_ocr": [True, "false"],
            "ocr_success": [pd.NA, "TRUE"],
            "math_enriched": [pd.NA, False],
            "is_empty": ["False", None],
        }
    )
    schema = ParquetSchema({"url_column": "url"})
    schema.write_metadata_parquet(df, parquet_path)

    out = pd.read_parquet(parquet_path)
    for column in ["needs_ocr", "ocr_success", "math_enriched", "enriched_math", "is_empty"]:
        assert out[column].dtype == pd.BooleanDtype()
    second = out.loc[out["filename"] == "doc2.pdf"].iloc[0]
    assert bool(second["needs_ocr"]) is False
    assert bool(second["ocr_success"]) is True


def test_math_enriched_alias_populated(tmp_path):
    parquet_path = tmp_path / "math.parquet"
    df = pd.DataFrame({"filename": ["stem.pdf"], "enriched_math": [True]})
    schema = ParquetSchema()
    schema.write_metadata_parquet(df, parquet_path)

    out = pd.read_parquet(parquet_path)
    row = out.iloc[0]
    assert bool(row["math_enriched"])
    assert bool(row["enriched_math"])


def test_nullable_or_handles_numpy_bool():
    import numpy as np

    df = pd.DataFrame(
        {
            "filename": ["stem.pdf"],
            "math_enriched": [np.bool_(True)],
            "enriched_math": [pd.NA],
        }
    )
    schema = ParquetSchema()
    normalised = schema.normalize_metadata_frame(df)
    row = normalised.iloc[0]
    assert bool(row["math_enriched"])
    assert bool(row["enriched_math"])


def test_ensure_metadata_parquet_no_artifacts(tmp_path, caplog):
    base_dir = tmp_path / "empty"
    base_dir.mkdir()
    schema = ParquetSchema({"url_column": "url"})
    with caplog.at_level("INFO"):
        result = schema.ensure_metadata_parquet(base_dir)
    assert result is None
    assert any("Unable to synthesise metadata parquet" in rec.message for rec in caplog.records)


def test_canonical_stem_variants():
    cases = {
        "foo.pdf": "foo",
        "bar.docling.json": "bar",
        "baz.docling.json.zst": "baz",
        "alpha.latex_map.jsonl": "alpha",
        "beta.metrics.json": "beta",
        "gamma.per_page.metrics.json": "gamma",
        "delta.with.dots.pdf": "delta.with.dots",
    }
    for source, expected in cases.items():
        assert canonical_stem(source) == expected


def test_corpus_reuses_canonical_metadata(tmp_path):
    from glossapi.corpus import Corpus

    root = tmp_path / "bundle"
    download_dir = root / "download_results"
    download_dir.mkdir(parents=True, exist_ok=True)

    schema = ParquetSchema({"url_column": "url"})
    canonical = download_dir / "download_results.parquet"
    df_canonical = pd.DataFrame(
        {
            "filename": ["doc.pdf"],
            "url": ["http://example.com/doc.pdf"],
            "math_enriched": [True],
        }
    )
    schema.write_metadata_parquet(df_canonical, canonical)

    metrics = download_dir / "download_results_remaining.parquet"
    df_metrics = pd.DataFrame(
        {
            "filename": ["doc.pdf"],
            "filter": ["ok"],
            "mojibake_badness_score": [0.0],
        }
    )
    df_metrics.to_parquet(metrics, index=False)

    corpus = Corpus(input_dir=root, output_dir=root)

    helper = ParquetSchema({"url_column": corpus.url_column})
    resolved = corpus._resolve_metadata_parquet(helper, ensure=True, search_input=True)
    assert resolved == canonical

    # Subsequent calls should reuse cached path without re-reading metrics file
    again = corpus._resolve_metadata_parquet(helper, ensure=False, search_input=False)
    assert again == canonical
