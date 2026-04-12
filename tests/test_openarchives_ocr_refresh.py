from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import zstandard as zstd

from glossapi.scripts.openarchives_ocr_refresh import (
    QUALITY_METHOD,
    _process_combined_ocr_dual_document_job_safe,
    _run_local_dual_document_job_safe,
    build_target_manifest,
    patch_openarchives_dataset,
    validate_patched_dataset,
)


def _write_jsonl_zst(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    with path.open("wb") as fh, cctx.stream_writer(fh) as writer:
        text = io.TextIOWrapper(writer, encoding="utf-8")
        for row in rows:
            text.write(json.dumps(row, ensure_ascii=False))
            text.write("\n")
        text.flush()


def _read_jsonl_zst(path: Path) -> list[dict]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        return [json.loads(line) for line in text if line.strip()]


def test_build_target_manifest_uses_relocated_merged_markdown_root(tmp_path: Path) -> None:
    hf_root = tmp_path / "hf"
    shard_path = hf_root / "data" / "openarchives" / "shard_001" / "shard_001-000000.jsonl.zst"
    rows = [
        {
            "doc_id": "doc-1",
            "filename": "AAA_001",
            "chunk_id": 0,
            "text": "original",
            "source_metadata": {},
            "pipeline_metadata": {"ocr_success": True, "needs_ocr": False},
        },
        {
            "doc_id": "doc-2",
            "filename": "BBB_001",
            "chunk_id": 0,
            "text": "untouched",
            "source_metadata": {},
            "pipeline_metadata": {"ocr_success": False, "needs_ocr": False},
        },
    ]
    _write_jsonl_zst(shard_path, rows)

    merged_root = tmp_path / "merged_markdown"
    merged_root.mkdir(parents=True, exist_ok=True)
    (merged_root / "AAA_001.md").write_text("clean me\n", encoding="utf-8")

    manifest_path = tmp_path / "ocr_manifest.parquet"
    pd.DataFrame(
        [
            {
                "source_doc_id": "doc-1",
                "filename_base": "AAA_001",
                "source_jsonl": "/irrelevant/source.jsonl.zst",
                "lane": "eu_node00_full_v1",
                "pages_total": 1,
                "page_count_merged": 1,
                "was_split": False,
                "split_parts": "",
                "merged_path": "/wrong/absolute/path/AAA_001.md",
                "text_sha256": "abc123",
            }
        ]
    ).to_parquet(manifest_path, index=False)

    run_root = tmp_path / "run"
    target_df = build_target_manifest(hf_root, manifest_path, run_root, merged_markdown_root=merged_root)

    assert list(target_df["doc_id"]) == ["doc-1"]
    assert target_df.iloc[0]["merged_path"] == str(merged_root / "AAA_001.md")
    summary = json.loads((run_root / "target_summary.json").read_text(encoding="utf-8"))
    assert summary["target_row_count"] == 1
    assert summary["merged_markdown_root"] == str(merged_root)


def test_patch_and_validate_dataset_updates_only_target_rows(tmp_path: Path) -> None:
    hf_root = tmp_path / "hf"
    shard_path = hf_root / "data" / "openarchives" / "shard_001" / "shard_001-000000.jsonl.zst"
    original_rows = [
        {
            "doc_id": "doc-1",
            "filename": "AAA_001",
            "chunk_id": 0,
            "text": "raw OCR text",
            "source_metadata": {"title": "A"},
            "pipeline_metadata": {
                "ocr_success": True,
                "needs_ocr": False,
                "filter": "ok",
                "percentage_greek": 12.0,
                "latin_percentage": 20.0,
                "polytonic_ratio": 0.0,
                "greek_badness_score": 80.0,
                "char_count_no_comments": 10,
                "is_empty": False,
                "page_count": 1,
                "pages_total": 1,
                "processing_stage": "old",
            },
        },
        {
            "doc_id": "doc-2",
            "filename": "BBB_001",
            "chunk_id": 0,
            "text": "leave me alone",
            "source_metadata": {"title": "B"},
            "pipeline_metadata": {
                "ocr_success": False,
                "needs_ocr": False,
                "filter": "ok",
                "processing_stage": "old",
            },
        },
    ]
    _write_jsonl_zst(shard_path, original_rows)

    target_df = pd.DataFrame(
        [
            {
                "doc_id": "doc-1",
                "filename": "AAA_001",
                "filename_base": "AAA_001",
                "lane": "eu_node00_full_v1",
                "merged_path": str(tmp_path / "merged_markdown" / "AAA_001.md"),
                "source_jsonl": str(shard_path),
                "pages_total": 3,
                "page_count_merged": 3,
                "text_sha256": "unused",
                "shard_relpath": "data/openarchives/shard_001/shard_001-000000.jsonl.zst",
            }
        ]
    )
    metrics_df = pd.DataFrame(
        [
            {
                "doc_id": "doc-1",
                "percentage_greek": 93.5,
                "latin_percentage": 0.5,
                "polytonic_ratio": 0.1,
                "greek_badness_score": 5.25,
                "char_count_no_comments": 321,
                "is_empty": False,
                "quality_method": QUALITY_METHOD,
                "reevaluated_at": "2026-04-12T10:00:00+00:00",
                "filter": "ok",
                "ocr_noise_suspect": False,
                "ocr_noise_flags": "",
                "ocr_repeat_phrase_run_max": 0,
                "ocr_repeat_line_run_max": 0,
                "ocr_repeat_suspicious_line_count": 0,
                "ocr_repeat_suspicious_line_ratio": 0.0,
            }
        ]
    )

    clean_dir = tmp_path / "clean_markdown"
    clean_dir.mkdir(parents=True, exist_ok=True)
    (clean_dir / "doc-1.md").write_text("cleaned text\n", encoding="utf-8")

    run_root = tmp_path / "run"
    patch_summary = patch_openarchives_dataset(hf_root, target_df, metrics_df, clean_dir, run_root)
    staged_root = Path(patch_summary["staged_root"])
    staged_rows = _read_jsonl_zst(staged_root / "data" / "openarchives" / "shard_001" / "shard_001-000000.jsonl.zst")

    updated = next(row for row in staged_rows if row["doc_id"] == "doc-1")
    untouched = next(row for row in staged_rows if row["doc_id"] == "doc-2")

    assert updated["text"] == "cleaned text"
    assert updated["pipeline_metadata"]["ocr_success"] is True
    assert updated["pipeline_metadata"]["needs_ocr"] is False
    assert updated["pipeline_metadata"]["page_count"] == 3
    assert updated["pipeline_metadata"]["pages_total"] == 3
    assert updated["pipeline_metadata"]["quality_method"] == QUALITY_METHOD
    assert updated["pipeline_metadata"]["processing_stage"] == "old"

    assert untouched == original_rows[1]

    validation = validate_patched_dataset(staged_root, target_df, clean_dir, run_root)
    assert validation["updated_rows_seen"] == 1
    assert validation["matched_text_sha_ok"] == 1


def test_process_pool_safe_wrapper_serializes_baseexception(monkeypatch: Any) -> None:
    def _boom(_job: tuple[str, ...]) -> dict:
        raise KeyboardInterrupt("panic-like failure")

    monkeypatch.setattr(
        "glossapi.scripts.openarchives_ocr_refresh._process_combined_ocr_dual_document_job",
        _boom,
    )

    result = _process_combined_ocr_dual_document_job_safe(
        ("input/doc-1.md", "clean/doc-1.md", "debug/doc-1.md", 10, 8, 10, 4, 3, 96)
    )

    assert result["ok"] is False
    assert result["source_stem"] == "doc-1"
    assert result["stage"] == "process_pool_worker"
    assert result["error_type"] == "KeyboardInterrupt"
    assert "panic-like failure" in result["error_message"]


def test_local_safe_wrapper_serializes_baseexception(monkeypatch: Any) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> dict:
        raise KeyboardInterrupt("local retry failure")

    monkeypatch.setattr(
        "glossapi.scripts.openarchives_ocr_refresh._process_combined_ocr_document",
        _boom,
    )

    result = _run_local_dual_document_job_safe(
        ("input/doc-2.md", "clean/doc-2.md", "debug/doc-2.md", 10, 8, 10, 4, 3, 96),
        noise_mod=object(),
    )

    assert result["ok"] is False
    assert result["source_stem"] == "doc-2"
    assert result["stage"] == "local_retry"
    assert result["error_type"] == "KeyboardInterrupt"
    assert "local retry failure" in result["error_message"]
