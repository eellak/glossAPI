from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import zstandard as zstd

from glossapi.scripts.openarchives_hf_refresh import main


def _write_jsonl_zst(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    with path.open("wb") as fh:
        with cctx.stream_writer(fh) as writer:
            for row in rows:
                writer.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def _read_jsonl_zst(path: Path) -> list[dict]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8").read()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_openarchives_hf_refresh_updates_pipeline_metadata_and_readme(tmp_path: Path) -> None:
    dataset_root = tmp_path / "openarchives.gr"
    shard_path = dataset_root / "data" / "openarchives" / "shard_001" / "chunk-000.jsonl.zst"
    _write_jsonl_zst(
        shard_path,
        [
            {
                "doc_id": "doc-a",
                "filename": "AAA_000",
                "text": "alpha",
                "source_metadata": {"filename": "AAA_000.pdf"},
                "pipeline_metadata": {"needs_ocr": False, "greek_badness_score": 1.0},
            },
            {
                "doc_id": "doc-b",
                "filename": "BBB_000",
                "text": "beta",
                "source_metadata": {"filename": "BBB_000.pdf"},
                "pipeline_metadata": {"needs_ocr": False, "greek_badness_score": 2.0},
            },
        ],
    )
    (dataset_root / "README.md").write_text(
        "---\npretty_name: OpenArchives.gr 191,000 docs\n---\n\n# OpenArchives.gr 191,000 docs\n\n"
        "- Σύνολο markdown αρχείων: **191,301** from openarchives.gr\n"
        "- Τα χαμηλής ποιότητας αρχεία που ενδέχεται να χρειάζονται OCR επεξεργασία επισημαίνονται με τη στήλη `needs_ocr`: **23,083 / 191,301 (12.07%)**\n"
        "- Total markdown files: **191,301** from openarchives.gr\n"
        "- Lower-quality files that may require OCR reprocessing are marked by the `needs_ocr` indicator: **23,083 / 191,301 (12.07%)**\n",
        encoding="utf-8",
    )

    metadata = tmp_path / "filled_document_level.parquet"
    pd.DataFrame(
        [
            {
                "source_doc_id": "doc-a",
                "source_jsonl": str(shard_path),
                "needs_ocr": True,
                "ocr_success": False,
                "greek_badness_score": 72.0,
                "mojibake_badness_score": 0.2,
                "latin_percentage": 33.3,
                "polytonic_ratio": 0.0,
                "char_count_no_comments": 1234.0,
                "is_empty": False,
                "filter": "ok",
                "quality_method": "refresh",
                "reevaluated_at": "2026-03-31T12:00:00+00:00",
            },
            {
                "source_doc_id": "doc-b",
                "source_jsonl": str(shard_path),
                "needs_ocr": False,
                "ocr_success": False,
                "greek_badness_score": 2.0,
                "mojibake_badness_score": 0.0,
                "latin_percentage": 22.0,
                "polytonic_ratio": 0.0,
                "char_count_no_comments": 456.0,
                "is_empty": True,
                "filter": "empty_text==0",
                "quality_method": "refresh",
                "reevaluated_at": "2026-03-31T12:00:00+00:00",
            },
        ]
    ).to_parquet(metadata, index=False)

    out_root = tmp_path / "out"
    rc = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--metadata-parquet",
            str(metadata),
            "--output-root",
            str(out_root),
        ]
    )
    assert rc == 0

    rows = _read_jsonl_zst(out_root / "data" / "openarchives" / "shard_001" / "chunk-000.jsonl.zst")
    assert rows[0]["pipeline_metadata"]["needs_ocr"] is True
    assert rows[0]["pipeline_metadata"]["greek_badness_score"] == 72.0
    assert rows[1]["pipeline_metadata"]["is_empty"] is True
    assert rows[1]["pipeline_metadata"]["filter"] == "empty_text==0"

    readme = (out_root / "README.md").read_text(encoding="utf-8")
    assert "OpenArchives.gr 2 docs" in readme
    assert "**1 / 2 (50.00%)**" in readme


def test_openarchives_hf_refresh_dry_run_does_not_write_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "openarchives.gr"
    shard_path = dataset_root / "data" / "openarchives" / "shard_001" / "chunk-000.jsonl.zst"
    _write_jsonl_zst(
        shard_path,
        [
            {
                "doc_id": "doc-a",
                "filename": "AAA_000",
                "text": "alpha",
                "source_metadata": {},
                "pipeline_metadata": {"needs_ocr": False},
            }
        ],
    )
    (dataset_root / "README.md").write_text("# OpenArchives.gr 191,000 docs\n", encoding="utf-8")
    metadata = tmp_path / "filled_document_level.parquet"
    pd.DataFrame(
        [
            {
                "source_doc_id": "doc-a",
                "source_jsonl": str(shard_path),
                "needs_ocr": True,
            }
        ]
    ).to_parquet(metadata, index=False)

    out_root = tmp_path / "out"
    rc = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--metadata-parquet",
            str(metadata),
            "--output-root",
            str(out_root),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not (out_root / "data" / "openarchives" / "shard_001" / "chunk-000.jsonl.zst").exists()
