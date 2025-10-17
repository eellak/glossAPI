import json
from pathlib import Path

import pandas as pd
import pytest

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
