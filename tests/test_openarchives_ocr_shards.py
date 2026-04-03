from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from glossapi.scripts import openarchives_ocr_cutoff_shards, openarchives_ocr_merge, openarchives_ocr_shards


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


def test_openarchives_ocr_cutoff_shards_uses_only_available_local_pdfs(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"source_doc_id": "doc-1", "filename": "a.html", "filename_base": "A", "needs_ocr": True, "pages_total_source": 100},
            {"source_doc_id": "doc-2", "filename": "b.html", "filename_base": "B", "needs_ocr": True, "pages_total_source": 50},
            {"source_doc_id": "doc-3", "filename": "c.html", "filename_base": "C", "needs_ocr": False, "pages_total_source": 999},
        ]
    )
    source = tmp_path / "master.parquet"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "A.pdf").write_bytes(b"%PDF-1.4\n")
    df.to_parquet(source, index=False)

    out_dir = tmp_path / "cutoff"
    rc = openarchives_ocr_cutoff_shards.main(
        [
            "--parquet",
            str(source),
            "--output-dir",
            str(out_dir),
            "--local-download-root",
            str(downloads),
            "--nodes",
            "2",
            "--cutoff-id",
            "cutoff-x",
        ]
    )
    assert rc == 0
    summary = json.loads((out_dir / "openarchives_ocr_cutoff_summary.json").read_text())
    assert summary["available_docs_total"] == 1
    assert summary["missing_docs_total"] == 1
    shard = pd.read_parquet(out_dir / "openarchives_ocr_shard_node_00.parquet")
    assert shard.loc[0, "source_filename"] == "a.html"
    assert shard.loc[0, "filename"] == "A.pdf"
    assert shard.loc[0, "md_filename"] == "A.md"
    assert bool(shard.loc[0, "available_at_cutoff"]) is True
    missing = pd.read_parquet(out_dir / "openarchives_ocr_missing_at_cutoff.parquet")
    assert set(missing["source_doc_id"]) == {"doc-2"}


def test_openarchives_ocr_merge_copies_markdown_artifacts(tmp_path: Path) -> None:
    master = pd.DataFrame(
        [
            {"source_doc_id": "doc-1", "filename": "a.html", "md_filename": "a.md", "needs_ocr": True, "ocr_success": False},
        ]
    )
    shard = pd.DataFrame(
        [
            {
                "source_doc_id": "doc-1",
                "filename": "A.pdf",
                "filename_base": "A",
                "md_filename": "A.md",
                "needs_ocr": False,
                "ocr_success": True,
            },
        ]
    )
    master_path = tmp_path / "master.parquet"
    shard_path = tmp_path / "shard.parquet"
    out_path = tmp_path / "merged.parquet"
    work_root = tmp_path / "node00"
    (work_root / "markdown").mkdir(parents=True)
    (work_root / "json" / "metrics").mkdir(parents=True)
    (work_root / "markdown" / "A.md").write_text("ocr text", encoding="utf-8")
    (work_root / "json" / "metrics" / "A.metrics.json").write_text("{}", encoding="utf-8")
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
            "--key-column",
            "source_doc_id",
            "--preserve-master-columns",
            "filename,md_filename",
            "--artifact-work-roots",
            str(work_root),
            "--artifact-output-root",
            str(tmp_path / "final"),
        ]
    )
    assert rc == 0
    merged = pd.read_parquet(out_path).set_index("source_doc_id")
    assert merged.loc["doc-1", "filename"] == "a.html"
    assert bool(merged.loc["doc-1", "ocr_success"]) is True
    assert merged.loc["doc-1", "text"] == "ocr text"
    assert merged.loc["doc-1", "ocr_markdown_relpath"] == "markdown/A.md"
    assert merged.loc["doc-1", "ocr_metrics_relpath"] == "json/metrics/A.metrics.json"
    assert merged.loc["doc-1", "ocr_text_sha256"] == hashlib.sha256(b"ocr text").hexdigest()
    assert (tmp_path / "final" / "markdown" / "A.md").exists()
    assert (tmp_path / "final" / "json" / "metrics" / "A.metrics.json").exists()


def test_openarchives_ocr_merge_embeds_text_without_copy_root(tmp_path: Path) -> None:
    master = pd.DataFrame(
        [
            {"source_doc_id": "doc-1", "filename": "a.html", "needs_ocr": True, "ocr_success": False},
        ]
    )
    shard = pd.DataFrame(
        [
            {
                "source_doc_id": "doc-1",
                "filename": "A.pdf",
                "filename_base": "A",
                "md_filename": "A.md",
                "needs_ocr": False,
                "ocr_success": True,
            },
        ]
    )
    master_path = tmp_path / "master.parquet"
    shard_path = tmp_path / "shard.parquet"
    out_path = tmp_path / "merged.parquet"
    work_root = tmp_path / "node00"
    (work_root / "markdown").mkdir(parents=True)
    (work_root / "json" / "metrics").mkdir(parents=True)
    (work_root / "markdown" / "A.md").write_text("embedded text", encoding="utf-8")
    (work_root / "json" / "metrics" / "A.metrics.json").write_text("{}", encoding="utf-8")
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
            "--key-column",
            "source_doc_id",
            "--artifact-work-roots",
            str(work_root),
        ]
    )
    assert rc == 0

    merged = pd.read_parquet(out_path).set_index("source_doc_id")
    assert merged.loc["doc-1", "text"] == "embedded text"
    assert pd.isna(merged.loc["doc-1", "ocr_markdown_relpath"])


def test_openarchives_ocr_merge_unifies_markdown_shards(tmp_path: Path) -> None:
    master = pd.DataFrame(
        [
            {"source_doc_id": "doc-1", "filename": "a.html", "md_filename": "a.md", "needs_ocr": True, "ocr_success": False},
        ]
    )
    shard = pd.DataFrame(
        [
            {
                "source_doc_id": "doc-1",
                "filename": "A.pdf",
                "filename_base": "A",
                "md_filename": "A.md",
                "needs_ocr": False,
                "ocr_success": True,
            },
        ]
    )
    master_path = tmp_path / "master.parquet"
    shard_path = tmp_path / "shard.parquet"
    out_path = tmp_path / "merged.parquet"
    work_root = tmp_path / "node00"
    markdown_dir = work_root / "markdown"
    markdown_dir.mkdir(parents=True)
    (markdown_dir / "A__p00001-00096.md").write_text("part one", encoding="utf-8")
    (markdown_dir / "A__p00097-00179.md").write_text("part two\n", encoding="utf-8")
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
            "--key-column",
            "source_doc_id",
            "--artifact-work-roots",
            str(work_root),
            "--artifact-output-root",
            str(tmp_path / "final"),
        ]
    )
    assert rc == 0

    merged = pd.read_parquet(out_path).set_index("source_doc_id")
    assert merged.loc["doc-1", "text"] == "part one\npart two\n"
    assert merged.loc["doc-1", "ocr_markdown_relpath"] == "markdown/A.md"
    assert (tmp_path / "final" / "markdown" / "A.md").read_text(encoding="utf-8") == "part one\npart two\n"
    assert (tmp_path / "final" / "sidecars" / "ocr_shards" / "markdown" / "A__p00001-00096.md").exists()
    assert (tmp_path / "final" / "sidecars" / "ocr_shards" / "markdown" / "A__p00097-00179.md").exists()
