from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from glossapi.scripts import openarchives_ocr_merge, openarchives_ocr_shards


def test_openarchives_ocr_shards_balances_pages(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"filename": "a.pdf", "needs_ocr": True, "pages_total": 100},
            {"filename": "b.pdf", "needs_ocr": True, "pages_total": 90},
            {"filename": "c.pdf", "needs_ocr": True, "pages_total": 40},
            {"filename": "d.pdf", "needs_ocr": True, "pages_total": 30},
            {"filename": "skip.pdf", "needs_ocr": False, "pages_total": 999},
        ]
    )
    source = tmp_path / "download_results.parquet"
    out_dir = tmp_path / "shards"
    df.to_parquet(source, index=False)

    rc = openarchives_ocr_shards.main(
        [
            "--parquet",
            str(source),
            "--output-dir",
            str(out_dir),
            "--nodes",
            "2",
        ]
    )
    assert rc == 0

    summary = json.loads((out_dir / "openarchives_ocr_shard_summary.json").read_text())
    assert summary["docs_total"] == 4
    assert summary["pages_total"] == 260
    manifests = sorted(out_dir.glob("openarchives_ocr_shard_node_*.parquet"))
    assert len(manifests) == 2
    page_totals = [int(pd.read_parquet(path)["pages_total"].sum()) for path in manifests]
    assert max(page_totals) - min(page_totals) <= 20


def test_openarchives_ocr_merge_updates_master(tmp_path: Path) -> None:
    master = pd.DataFrame(
        [
            {"filename": "a.pdf", "needs_ocr": True, "ocr_success": False},
            {"filename": "b.pdf", "needs_ocr": True, "ocr_success": False},
        ]
    )
    shard = pd.DataFrame(
        [
            {"filename": "a.pdf", "needs_ocr": False, "ocr_success": True, "ocr_node_id": 2},
        ]
    )
    master_path = tmp_path / "master.parquet"
    shard_path = tmp_path / "shard.parquet"
    out_path = tmp_path / "merged.parquet"
    master.to_parquet(master_path, index=False)
    shard.to_parquet(shard_path, index=False)

    rc = openarchives_ocr_merge.main(
        [
            "--master-parquet",
            str(master_path),
            "--shard-parquets",
            str(shard_path),
            "--output-parquet",
            str(out_path),
        ]
    )
    assert rc == 0

    merged = pd.read_parquet(out_path).set_index("filename")
    assert bool(merged.loc["a.pdf", "ocr_success"]) is True
    assert bool(merged.loc["a.pdf", "needs_ocr"]) is False
    assert int(merged.loc["a.pdf", "ocr_node_id"]) == 2
    assert bool(merged.loc["b.pdf", "ocr_success"]) is False
