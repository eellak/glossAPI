from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import zstandard as zstd

from glossapi.scripts.openarchives_ocr_enrich import main


def _write_jsonl_zst(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows).encode("utf-8")
    cctx = zstd.ZstdCompressor()
    path.write_bytes(cctx.compress(payload))


def test_openarchives_ocr_enrich_extracts_page_counts_and_pdf_url(tmp_path):
    raw_root = tmp_path / "raw" / "openarchives.gr"
    jsonl_path = raw_root / "data" / "openarchives" / "shard_01" / "chunk-000.jsonl.zst"
    _write_jsonl_zst(
        jsonl_path,
        [
            {
                "doc_id": "doc-a",
                "filename": "AAA_000",
                "text": "alpha",
                "pipeline_metadata": {"page_count": 98, "pages_total": 98},
                "source_metadata": {
                    "pdf_links_json": "https://example.com/a.pdf",
                    "collection_slug": "Dione",
                    "language_code": "el",
                },
            },
            {
                "doc_id": "doc-b",
                "filename": "BBB_000",
                "text": "beta",
                "pipeline_metadata": {"pages_total": 12},
                "source_metadata": {
                    "pdf_links_json": json.dumps(
                        [
                            {"url": "https://example.com/b.pdf"},
                            {"url": "https://example.com/b2.pdf"},
                        ]
                    ),
                    "collection_slug": "Pandemos",
                    "language_code": "el",
                },
            },
        ],
    )

    parquet = tmp_path / "document_level.parquet"
    pd.DataFrame(
        [
            {
                "source_doc_id": "doc-a",
                "filename": "AAA_000.pdf",
                "source_jsonl": str(jsonl_path),
                "needs_ocr": True,
            },
            {
                "source_doc_id": "doc-b",
                "filename": "BBB_000.pdf",
                "source_jsonl": str(jsonl_path),
                "needs_ocr": True,
            },
            {
                "source_doc_id": "doc-c",
                "filename": "CCC_000.pdf",
                "source_jsonl": str(jsonl_path),
                "needs_ocr": False,
            },
        ]
    ).to_parquet(parquet, index=False)

    output = tmp_path / "enriched.parquet"
    rc = main(
        [
            "--parquet",
            str(parquet),
            "--raw-repo-root",
            str(raw_root),
            "--output-parquet",
            str(output),
        ]
    )
    assert rc == 0

    enriched = pd.read_parquet(output).sort_values("filename").reset_index(drop=True)
    assert enriched["filename"].tolist() == ["AAA_000.pdf", "BBB_000.pdf"]
    assert enriched["page_count_source"].tolist() == [98, 12]
    assert enriched["pages_total_source"].tolist() == [98, 12]
    assert enriched["pdf_url"].tolist() == ["https://example.com/a.pdf", "https://example.com/b.pdf"]
    assert enriched["source_collection_slug"].tolist() == ["Dione", "Pandemos"]


def test_openarchives_ocr_enrich_resolves_rewritten_source_jsonl_path(tmp_path):
    raw_root = tmp_path / "raw" / "openarchives.gr"
    jsonl_path = raw_root / "data" / "openarchives" / "shard_02" / "chunk-001.jsonl.zst"
    _write_jsonl_zst(
        jsonl_path,
        [
            {
                "doc_id": "doc-x",
                "filename": "XXX_000",
                "text": "x",
                "pipeline_metadata": {"page_count": 7},
                "source_metadata": {"external_link": "https://example.com/x"},
            }
        ],
    )

    parquet = tmp_path / "document_level.parquet"
    pd.DataFrame(
        [
            {
                "source_doc_id": "doc-x",
                "filename": "XXX_000.pdf",
                "source_jsonl": "/home/foivos/data/glossapi_raw/hf/openarchives.gr/data/openarchives/shard_02/chunk-001.jsonl.zst",
                "needs_ocr": True,
            }
        ]
    ).to_parquet(parquet, index=False)

    output = tmp_path / "enriched.parquet"
    rc = main(
        [
            "--parquet",
            str(parquet),
            "--raw-repo-root",
            str(raw_root),
            "--output-parquet",
            str(output),
        ]
    )
    assert rc == 0

    enriched = pd.read_parquet(output)
    assert int(enriched.loc[0, "page_count_source"]) == 7
    assert enriched.loc[0, "pdf_url"] == "https://example.com/x"
