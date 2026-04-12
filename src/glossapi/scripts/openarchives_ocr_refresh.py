from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import logging
import math
import os
import re
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd
import zstandard as zstd
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import get_token

from glossapi.corpus.phase_clean import (
    PAGE_SPLIT_MARKER,
    _can_use_combined_ocr_process_pool,
    _combined_ocr_process_pool_warning_ctx,
    _init_combined_ocr_worker,
    _process_combined_ocr_document,
    _process_combined_ocr_dual_document_job,
    _summarize_metric,
)
from glossapi.corpus.text_surface_metrics import sanitized_char_count

LOGGER = logging.getLogger("openarchives_ocr_refresh")
DEFAULT_REPO_ID = "glossAPI/openarchives.gr"
QUALITY_METHOD = "glossapi_clean_ocr_refresh_v1"
RENDER_REQUIRED_ATTRS = (
    "find_numeric_debug_page_spans",
    "evaluate_page_character_noise",
    "score_markdown_directory_ocr_profile",
    "score_markdown_directory_detailed",
)


@dataclass(frozen=True)
class TargetDoc:
    doc_id: str
    filename_base: str
    lane: str
    merged_path: Path
    source_jsonl: str
    pages_total: Optional[int]
    page_count_merged: int
    text_sha256: str
    shard_relpath: str


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh OpenArchives OCR-success rows with cleaned texts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-target-manifest")
    build.add_argument("--hf-root", type=Path, required=True)
    build.add_argument("--ocr-manifest", type=Path, required=True)
    build.add_argument("--run-root", type=Path, required=True)
    build.add_argument("--merged-markdown-root", type=Path, default=None)
    build.add_argument("--verbose", action="store_true")

    clean = subparsers.add_parser("run-clean")
    clean.add_argument("--target-manifest", type=Path, required=True)
    clean.add_argument("--run-root", type=Path, required=True)
    clean.add_argument("--render-workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    clean.add_argument("--min-progress-steps", type=int, default=10)
    clean.add_argument("--min-repeat-steps", type=int, default=8)
    clean.add_argument("--min-same-digit-steps", type=int, default=10)
    clean.add_argument("--word-rep-threshold", type=int, default=4)
    clean.add_argument("--word-min-period", type=int, default=3)
    clean.add_argument("--word-window", type=int, default=96)
    clean.add_argument("--verbose", action="store_true")

    metrics = subparsers.add_parser("reevaluate-cleaned")
    metrics.add_argument("--target-manifest", type=Path, required=True)
    metrics.add_argument("--clean-dir", type=Path, required=True)
    metrics.add_argument("--run-root", type=Path, required=True)
    metrics.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() or 1))
    metrics.add_argument("--min-repeat-run", type=int, default=6)
    metrics.add_argument("--verbose", action="store_true")

    patch = subparsers.add_parser("patch-dataset")
    patch.add_argument("--hf-root", type=Path, required=True)
    patch.add_argument("--target-manifest", type=Path, required=True)
    patch.add_argument("--clean-dir", type=Path, required=True)
    patch.add_argument("--metrics-parquet", type=Path, required=True)
    patch.add_argument("--run-root", type=Path, required=True)
    patch.add_argument("--verbose", action="store_true")

    validate = subparsers.add_parser("validate-patch")
    validate.add_argument("--staged-root", type=Path, required=True)
    validate.add_argument("--target-manifest", type=Path, required=True)
    validate.add_argument("--clean-dir", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--verbose", action="store_true")

    upload = subparsers.add_parser("upload")
    upload.add_argument("--staged-root", type=Path, required=True)
    upload.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    upload.add_argument("--revision", default=None)
    upload.add_argument("--num-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    upload.add_argument("--verbose", action="store_true")

    run = subparsers.add_parser("run-end-to-end")
    run.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    run.add_argument("--hf-root", type=Path, required=True)
    run.add_argument("--ocr-manifest", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--merged-markdown-root", type=Path, default=None)
    run.add_argument("--skip-download", action="store_true")
    run.add_argument("--revision", default=None)
    run.add_argument("--render-workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    run.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() or 1))
    run.add_argument("--upload-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    run.add_argument("--min-progress-steps", type=int, default=10)
    run.add_argument("--min-repeat-steps", type=int, default=8)
    run.add_argument("--min-same-digit-steps", type=int, default=10)
    run.add_argument("--word-rep-threshold", type=int, default=4)
    run.add_argument("--word-min-period", type=int, default=3)
    run.add_argument("--word-window", type=int, default=96)
    run.add_argument("--min-repeat-run", type=int, default=6)
    run.add_argument("--upload", action="store_true")
    run.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _serialize_render_job_failure(
    job: tuple[str, ...],
    exc: BaseException,
    *,
    stage: str,
) -> Dict[str, Any]:
    source_path = Path(job[0])
    return {
        "ok": False,
        "job": list(job),
        "source_path": str(source_path),
        "source_stem": source_path.stem,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _process_combined_ocr_dual_document_job_safe(
    job: tuple[str, str, str, int, int, int, int, int, int],
) -> Dict[str, Any]:
    try:
        return {"ok": True, "doc_result": _process_combined_ocr_dual_document_job(job)}
    except BaseException as exc:  # pragma: no cover - exercised on remote pathological docs
        return _serialize_render_job_failure(job, exc, stage="process_pool_worker")


def _run_local_dual_document_job_safe(
    job: tuple[str, str, str, int, int, int, int, int, int],
    *,
    noise_mod: Any,
) -> Dict[str, Any]:
    try:
        source_path_str, clean_output_path_str, debug_output_path_str, *_ = job
        return {
            "ok": True,
            "doc_result": _process_combined_ocr_document(
                Path(source_path_str),
                clean_output_path=Path(clean_output_path_str),
                debug_output_path=Path(debug_output_path_str),
                noise_mod=noise_mod,
                min_progress_steps=int(job[3]),
                min_repeat_steps=int(job[4]),
                min_same_digit_steps=int(job[5]),
                word_rep_threshold=int(job[6]),
                word_min_period=int(job[7]),
                word_window=int(job[8]),
                include_page_metrics=True,
                include_match_index=True,
            ),
        }
    except BaseException as exc:  # pragma: no cover - exercised on remote pathological docs
        return _serialize_render_job_failure(job, exc, stage="local_retry")


def _load_noise_mod() -> Any:
    try:
        noise_mod = importlib.import_module("glossapi_rs_noise")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "glossapi_rs_noise is not importable. Bootstrap the environment with maturin develop -m rust/glossapi_rs_noise/Cargo.toml first."
        ) from exc
    missing = [attr for attr in RENDER_REQUIRED_ATTRS if not hasattr(noise_mod, attr)]
    if missing:
        raise RuntimeError(f"glossapi_rs_noise missing required attrs: {missing}")
    return noise_mod


def _snapshot_download_hf_repo(repo_id: str, local_dir: Path, revision: Optional[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=revision,
        token=get_token(),
    )


def _stage_hf_repo(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)

    def _ignore(_dir: str, names: List[str]) -> set[str]:
        ignored: set[str] = set()
        if ".cache" in names:
            ignored.add(".cache")
        return ignored

    shutil.copytree(src_root, dst_root, copy_function=lambda s, d: _hardlink_or_copy(Path(s), Path(d)), ignore=_ignore)


def _zstd_read_lines(path: Path) -> Iterator[Dict[str, Any]]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:
            if line.strip():
                yield json.loads(line)


def _zstd_write_lines(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    cctx = zstd.ZstdCompressor(level=9)
    with tmp_path.open("wb") as fh, cctx.stream_writer(fh) as writer:
        text = io.TextIOWrapper(writer, encoding="utf-8")
        for row in rows:
            text.write(json.dumps(row, ensure_ascii=False))
            text.write("\n")
            count += 1
        text.flush()
    tmp_path.replace(path)
    return count


def _compute_filter(greek_badness_score: float, ocr_noise_suspect: bool) -> str:
    tokens: List[str] = []
    if greek_badness_score > 60.0:
        tokens.append("greek>60")
    if bool(ocr_noise_suspect):
        tokens.append("ocr_noise")
    return ";".join(tokens) if tokens else "ok"


def _load_target_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    if df.empty:
        raise RuntimeError(f"target manifest is empty: {path}")
    return df


def build_target_manifest(
    hf_root: Path,
    ocr_manifest_path: Path,
    run_root: Path,
    merged_markdown_root: Optional[Path] = None,
) -> pd.DataFrame:
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.read_parquet(ocr_manifest_path).copy()
    if manifest_df.empty:
        raise RuntimeError(f"OCR manifest is empty: {ocr_manifest_path}")
    docs_by_id: Dict[str, Dict[str, Any]] = {}
    for row in manifest_df.to_dict("records"):
        doc_id = str(row["source_doc_id"])
        if doc_id in docs_by_id:
            raise RuntimeError(f"duplicate source_doc_id in OCR manifest: {doc_id}")
        docs_by_id[doc_id] = row

    shard_root = hf_root / "data" / "openarchives"
    if not shard_root.exists():
        raise RuntimeError(f"OpenArchives shard root not found: {shard_root}")

    resolved_merged_root = merged_markdown_root
    if resolved_merged_root is None:
        inferred_root = ocr_manifest_path.parent / 'merged_markdown'
        if inferred_root.exists():
            resolved_merged_root = inferred_root
    rows: List[Dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for shard_path in sorted(shard_root.rglob("*.jsonl.zst")):
        shard_relpath = shard_path.relative_to(hf_root).as_posix()
        for obj in _zstd_read_lines(shard_path):
            pipeline_metadata = obj.get("pipeline_metadata") or {}
            if not bool(pipeline_metadata.get("ocr_success")):
                continue
            doc_id = str(obj.get("doc_id") or "")
            if not doc_id:
                raise RuntimeError(f"row in {shard_relpath} missing doc_id")
            if doc_id in seen_doc_ids:
                raise RuntimeError(f"duplicate doc_id in OpenArchives OCR-success set: {doc_id}")
            seen_doc_ids.add(doc_id)
            manifest_row = docs_by_id.get(doc_id)
            if manifest_row is None:
                raise RuntimeError(f"OpenArchives OCR-success doc missing from OCR manifest: {doc_id}")
            merged_path = Path(str(manifest_row['merged_path']))
            if resolved_merged_root is not None:
                merged_path = resolved_merged_root / f"{manifest_row['filename_base']}.md"
            if not merged_path.exists():
                raise RuntimeError(f"premerged OCR markdown missing for {doc_id}: {merged_path}")
            rows.append(
                {
                    "doc_id": doc_id,
                    "filename": str(obj.get("filename") or ""),
                    "filename_base": str(manifest_row["filename_base"]),
                    "lane": str(manifest_row["lane"]),
                    "merged_path": str(merged_path),
                    "source_jsonl": str(manifest_row["source_jsonl"]),
                    "pages_total": _safe_int(manifest_row.get("pages_total")),
                    "page_count_merged": int(manifest_row["page_count_merged"]),
                    "text_sha256": str(manifest_row["text_sha256"]),
                    "shard_relpath": shard_relpath,
                }
            )

    if not rows:
        raise RuntimeError("no OpenArchives OCR-success rows were found")
    target_df = pd.DataFrame(rows).sort_values(["shard_relpath", "doc_id"], kind="stable").reset_index(drop=True)
    manifest_path = run_root / "target_manifest.parquet"
    target_df.to_parquet(manifest_path, index=False)
    summary = {
        "target_row_count": int(len(target_df)),
        "target_manifest": str(manifest_path),
        "unique_shards": int(target_df["shard_relpath"].nunique()),
        "lanes": {str(k): int(v) for k, v in target_df['lane'].value_counts().sort_index().to_dict().items()},
        "merged_markdown_root": str(resolved_merged_root) if resolved_merged_root is not None else None,
    }
    (run_root / "target_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Built target manifest with %d OCR-success docs", len(target_df))
    return target_df


def materialize_target_markdown(target_df: pd.DataFrame, input_dir: Path) -> None:
    _ensure_clean_dir(input_dir)
    for row in target_df.to_dict("records"):
        src = Path(str(row["merged_path"]))
        dst = input_dir / f"{row['doc_id']}.md"
        _hardlink_or_copy(src, dst)


def run_fast_clean_debug(
    target_df: pd.DataFrame,
    run_root: Path,
    *,
    render_workers: int,
    min_progress_steps: int,
    min_repeat_steps: int,
    min_same_digit_steps: int,
    word_rep_threshold: int,
    word_min_period: int,
    word_window: int,
) -> Dict[str, Path]:
    noise_mod = _load_noise_mod()
    input_dir = run_root / "target_markdown"
    clean_dir = run_root / "clean_markdown"
    debug_dir = run_root / "debug"
    manifest_path = debug_dir / "manifest.jsonl"
    page_metrics_path = debug_dir / "page_metrics.jsonl"
    match_index_path = debug_dir / "match_index.jsonl"
    render_failures_path = debug_dir / "render_failures.jsonl"
    summary_path = debug_dir / "summary.json"

    materialize_target_markdown(target_df, input_dir)
    _ensure_clean_dir(clean_dir)
    _ensure_clean_dir(debug_dir)
    md_files = sorted(input_dir.glob("*.md"))

    rows: List[Dict[str, Any]] = []
    total_page_times: List[float] = []
    table_page_times: List[float] = []
    numeric_page_times: List[float] = []
    latex_page_times: List[float] = []
    hybrid_page_times: List[float] = []
    shared_page_times: List[float] = []
    char_eval_times: List[float] = []
    bad_char_ratios: List[float] = []

    failures: List[Dict[str, Any]] = []

    def _consume(doc_result: Dict[str, Any], page_metrics_handle: Any, match_index_handle: Any) -> None:
        row = dict(doc_result["row"])
        row["doc_id"] = row["source_stem"]
        rows.append(row)
        for page_row in doc_result["page_metric_rows"]:
            page_row = dict(page_row)
            page_row["doc_id"] = page_row["source_stem"]
            page_metrics_handle.write(json.dumps(page_row, ensure_ascii=False))
            page_metrics_handle.write("\n")
            total_page_times.append(float(page_row["total_page_seconds"]))
            table_page_times.append(float(page_row["table_seconds"]))
            numeric_page_times.append(float(page_row["numeric_seconds"]))
            latex_page_times.append(float(page_row["latex_seconds"]))
            hybrid_page_times.append(float(page_row["hybrid_seconds"]))
            shared_page_times.append(float(page_row["shared_repeat_seconds"]))
            char_eval_times.append(float(page_row["char_eval_seconds"]))
            bad_char_ratios.append(float(page_row["bad_char_ratio"]))
        for match_row in doc_result["match_index_rows"]:
            match_row = dict(match_row)
            match_row["doc_id"] = Path(str(match_row["source_path"])).stem
            match_index_handle.write(json.dumps(match_row, ensure_ascii=False))
            match_index_handle.write("\n")

    jobs = [
        (
            str(source_path),
            str(clean_dir / source_path.name),
            str(debug_dir / source_path.name),
            int(min_progress_steps),
            int(min_repeat_steps),
            int(min_same_digit_steps),
            int(word_rep_threshold),
            int(word_min_period),
            int(word_window),
        )
        for source_path in md_files
    ]

    with page_metrics_path.open("w", encoding="utf-8") as page_metrics_handle, match_index_path.open("w", encoding="utf-8") as match_index_handle:
        if _can_use_combined_ocr_process_pool(noise_mod, render_workers):
            with _combined_ocr_process_pool_warning_ctx():
                with ProcessPoolExecutor(max_workers=render_workers, initializer=_init_combined_ocr_worker) as executor:
                    for worker_result in executor.map(_process_combined_ocr_dual_document_job_safe, jobs):
                        if worker_result.get("ok"):
                            _consume(worker_result["doc_result"], page_metrics_handle, match_index_handle)
                            continue
                        source_path = str(worker_result.get("source_path", ""))
                        LOGGER.warning(
                            "Process-pool OCR render failed for %s; retrying locally (%s: %s)",
                            source_path,
                            worker_result.get("error_type"),
                            worker_result.get("error_message"),
                        )
                        local_result = _run_local_dual_document_job_safe(tuple(worker_result["job"]), noise_mod=noise_mod)
                        if local_result.get("ok"):
                            _consume(local_result["doc_result"], page_metrics_handle, match_index_handle)
                            continue
                        failures.append(dict(local_result))
        else:
            with ThreadPoolExecutor(max_workers=render_workers) as executor:
                for worker_result in executor.map(lambda job: _run_local_dual_document_job_safe(job, noise_mod=noise_mod), jobs):
                    if worker_result.get("ok"):
                        _consume(worker_result["doc_result"], page_metrics_handle, match_index_handle)
                    else:
                        failures.append(dict(worker_result))

    if failures:
        with render_failures_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False))
                handle.write("\n")
        first = failures[0]
        raise RuntimeError(
            "OCR render failed for "
            f"{first.get('source_stem')} during {first.get('stage')}: "
            f"{first.get('error_type')}: {first.get('error_message')} "
            f"(details in {render_failures_path})"
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "doc_count": len(rows),
        "matched_doc_count": sum(1 for row in rows if int(row["matched_page_count"]) > 0),
        "matched_page_count": int(sum(int(row["matched_page_count"]) for row in rows)),
        "match_count": int(sum(int(row.get("match_count", 0)) for row in rows)),
        "table_match_count": int(sum(int(row["table_match_count"]) for row in rows)),
        "numeric_match_count": int(sum(int(row["numeric_match_count"]) for row in rows)),
        "latex_match_count": int(sum(int(row["latex_match_count"]) for row in rows)),
        "hybrid_match_count": int(sum(int(row["hybrid_match_count"]) for row in rows)),
        "word_match_count": int(sum(int(row["word_match_count"]) for row in rows)),
        "word_rep_threshold": int(word_rep_threshold),
        "word_min_period": int(word_min_period),
        "word_window": int(word_window),
        "total_page_seconds": _summarize_metric(total_page_times),
        "table_seconds": _summarize_metric(table_page_times),
        "numeric_seconds": _summarize_metric(numeric_page_times),
        "latex_seconds": _summarize_metric(latex_page_times),
        "hybrid_seconds": _summarize_metric(hybrid_page_times),
        "shared_repeat_seconds": _summarize_metric(shared_page_times),
        "char_eval_seconds": _summarize_metric(char_eval_times),
        "bad_char_ratio": _summarize_metric(bad_char_ratios),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target_df[["doc_id", "filename_base", "lane", "shard_relpath"]].to_parquet(run_root / "clean_target_docs.parquet", index=False)
    LOGGER.info("Rendered clean+debug outputs for %d docs", len(rows))
    return {
        "input_dir": input_dir,
        "clean_dir": clean_dir,
        "debug_dir": debug_dir,
        "manifest_path": manifest_path,
        "page_metrics_path": page_metrics_path,
        "match_index_path": match_index_path,
        "summary_path": summary_path,
    }


def reevaluate_cleaned_outputs(
    target_df: pd.DataFrame,
    clean_dir: Path,
    run_root: Path,
    *,
    num_threads: int,
    min_repeat_run: int,
) -> pd.DataFrame:
    noise_mod = _load_noise_mod()
    profile_rows = list(noise_mod.score_markdown_directory_ocr_profile(str(clean_dir), int(num_threads), int(min_repeat_run)))
    if not profile_rows:
        raise RuntimeError(f"no OCR profile rows returned for {clean_dir}")
    df_profile = pd.DataFrame(profile_rows).copy()
    df_profile["doc_id"] = df_profile["path"].apply(lambda value: Path(str(value)).stem)

    detailed_rows = []
    for row in noise_mod.score_markdown_directory_detailed(str(clean_dir), int(num_threads)):
        detailed_rows.append(
            {
                "doc_id": Path(str(row[0])).stem,
                "greek_badness_score": float(row[1]),
                "detailed_latin_percentage": float(row[2]),
                "table_ratio": float(row[3]),
                "detailed_polytonic_ratio": float(row[4]),
                "len_greek": int(row[5]),
            }
        )
    df_detailed = pd.DataFrame(detailed_rows)

    char_rows: List[Dict[str, Any]] = []
    for clean_path in sorted(clean_dir.glob("*.md")):
        text = clean_path.read_text(encoding="utf-8")
        char_count_no_comments, is_empty = sanitized_char_count(text)
        char_rows.append(
            {
                "doc_id": clean_path.stem,
                "char_count_no_comments": int(char_count_no_comments),
                "is_empty": bool(is_empty),
            }
        )
    df_chars = pd.DataFrame(char_rows)

    merged = target_df[["doc_id", "pages_total", "page_count_merged"]].merge(df_profile, on="doc_id", how="left")
    merged = merged.merge(df_detailed, on="doc_id", how="left")
    merged = merged.merge(df_chars, on="doc_id", how="left")

    required_cols = [
        "percentage_greek",
        "latin_percentage",
        "polytonic_ratio",
        "greek_badness_score",
        "char_count_no_comments",
        "is_empty",
    ]
    merged["latin_percentage"] = merged["latin_percentage"].fillna(merged.get("detailed_latin_percentage"))
    merged["polytonic_ratio"] = merged["polytonic_ratio"].fillna(merged.get("detailed_polytonic_ratio"))
    missing = {}
    for col in required_cols:
        if merged[col].isna().any():
            missing[col] = int(merged[col].isna().sum())
    if missing:
        raise RuntimeError(f"reevaluated clean metrics missing required values: {missing}")

    merged["quality_method"] = QUALITY_METHOD
    reevaluated_at = datetime.now(UTC).isoformat()
    merged["reevaluated_at"] = reevaluated_at
    merged["filter"] = merged.apply(
        lambda row: _compute_filter(float(row["greek_badness_score"]), bool(row["ocr_noise_suspect"])),
        axis=1,
    )

    output_cols = [
        "doc_id",
        "percentage_greek",
        "latin_percentage",
        "polytonic_ratio",
        "greek_badness_score",
        "char_count_no_comments",
        "is_empty",
        "quality_method",
        "reevaluated_at",
        "filter",
        "ocr_noise_suspect",
        "ocr_noise_flags",
        "ocr_repeat_phrase_run_max",
        "ocr_repeat_line_run_max",
        "ocr_repeat_suspicious_line_count",
        "ocr_repeat_suspicious_line_ratio",
    ]
    metrics_df = merged[output_cols].copy()
    metrics_path = run_root / "clean_metrics.parquet"
    metrics_df.to_parquet(metrics_path, index=False)
    summary = {
        "metrics_rows": int(len(metrics_df)),
        "metrics_path": str(metrics_path),
        "quality_method": QUALITY_METHOD,
        "reevaluated_at": reevaluated_at,
    }
    (run_root / "clean_metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Reevaluated cleaned outputs for %d docs", len(metrics_df))
    return metrics_df


def _patch_pipeline_metadata(
    pipeline_metadata: Dict[str, Any],
    *,
    metric_row: Dict[str, Any],
    page_total: int,
) -> Dict[str, Any]:
    pm = dict(pipeline_metadata or {})
    pm["filter"] = str(metric_row["filter"])
    pm["needs_ocr"] = False
    pm["ocr_success"] = True
    pm["percentage_greek"] = float(metric_row["percentage_greek"])
    pm["latin_percentage"] = float(metric_row["latin_percentage"])
    pm["polytonic_ratio"] = float(metric_row["polytonic_ratio"])
    pm["greek_badness_score"] = float(metric_row["greek_badness_score"])
    pm["char_count_no_comments"] = int(metric_row["char_count_no_comments"])
    pm["is_empty"] = bool(metric_row["is_empty"])
    pm["quality_method"] = str(metric_row["quality_method"])
    pm["reevaluated_at"] = str(metric_row["reevaluated_at"])
    pm["page_count"] = int(page_total)
    pm["pages_total"] = int(page_total)
    pm["ocr_noise_suspect"] = bool(metric_row["ocr_noise_suspect"])
    pm["ocr_noise_flags"] = str(metric_row["ocr_noise_flags"] or "")
    pm["ocr_repeat_phrase_run_max"] = int(metric_row["ocr_repeat_phrase_run_max"])
    pm["ocr_repeat_line_run_max"] = int(metric_row["ocr_repeat_line_run_max"])
    pm["ocr_repeat_suspicious_line_count"] = int(metric_row["ocr_repeat_suspicious_line_count"])
    pm["ocr_repeat_suspicious_line_ratio"] = float(metric_row["ocr_repeat_suspicious_line_ratio"])
    return pm


def patch_openarchives_dataset(
    hf_root: Path,
    target_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    clean_dir: Path,
    run_root: Path,
) -> Dict[str, Any]:
    staged_root = run_root / "staged_openarchives"
    _stage_hf_repo(hf_root, staged_root)

    target_map = {str(row["doc_id"]): row for row in target_df.to_dict("records")}
    metric_map = {str(row["doc_id"]): row for row in metrics_df.to_dict("records")}
    updated_doc_ids: set[str] = set()
    total_rows = 0
    total_ocr_success_rows = 0
    touched_shards: List[str] = []

    shard_root = staged_root / "data" / "openarchives"
    for shard_path in sorted(shard_root.rglob("*.jsonl.zst")):
        shard_rel = shard_path.relative_to(staged_root).as_posix()
        shard_updated = False

        def _patched_rows() -> Iterator[Dict[str, Any]]:
            nonlocal total_rows, total_ocr_success_rows, shard_updated
            for obj in _zstd_read_lines(shard_path):
                total_rows += 1
                pm = obj.get("pipeline_metadata") or {}
                if bool(pm.get("ocr_success")):
                    total_ocr_success_rows += 1
                    doc_id = str(obj.get("doc_id") or "")
                    target = target_map.get(doc_id)
                    if target is None:
                        raise RuntimeError(f"OpenArchives OCR-success row missing from target manifest: {doc_id}")
                    metric_row = metric_map.get(doc_id)
                    if metric_row is None:
                        raise RuntimeError(f"clean metrics missing for target doc: {doc_id}")
                    clean_path = clean_dir / f"{doc_id}.md"
                    if not clean_path.exists():
                        raise RuntimeError(f"cleaned markdown missing for target doc: {clean_path}")
                    text = clean_path.read_text(encoding="utf-8")
                    obj["text"] = text.rstrip("\n")
                    page_total = _safe_int(target.get("pages_total")) or int(target["page_count_merged"])
                    obj["pipeline_metadata"] = _patch_pipeline_metadata(pm, metric_row=metric_row, page_total=int(page_total))
                    updated_doc_ids.add(doc_id)
                    shard_updated = True
                yield obj

        row_count = _zstd_write_lines(shard_path, _patched_rows())
        if shard_updated:
            touched_shards.append(f"{shard_rel}\t{row_count}")

    expected_doc_ids = set(target_map)
    if updated_doc_ids != expected_doc_ids:
        missing = sorted(expected_doc_ids - updated_doc_ids)[:20]
        extra = sorted(updated_doc_ids - expected_doc_ids)[:20]
        raise RuntimeError(f"patched doc_id set mismatch: missing={missing} extra={extra}")
    summary = {
        "staged_root": str(staged_root),
        "total_rows": int(total_rows),
        "ocr_success_rows_seen": int(total_ocr_success_rows),
        "updated_rows": int(len(updated_doc_ids)),
        "touched_shard_count": int(len(touched_shards)),
    }
    (run_root / "patched_touched_shards.tsv").write_text("\n".join(touched_shards) + "\n", encoding="utf-8")
    (run_root / "patch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Patched staged OpenArchives repo with %d cleaned rows", len(updated_doc_ids))
    return summary


def validate_patched_dataset(
    staged_root: Path,
    target_df: pd.DataFrame,
    clean_dir: Path,
    run_root: Path,
) -> Dict[str, Any]:
    target_doc_ids = set(str(v) for v in target_df["doc_id"].tolist())
    updated_rows_seen = 0
    matched_text_sha_ok = 0
    total_rows = 0
    ocr_success_true = 0
    for shard_path in sorted((staged_root / "data" / "openarchives").rglob("*.jsonl.zst")):
        for obj in _zstd_read_lines(shard_path):
            total_rows += 1
            pipeline_metadata = obj.get("pipeline_metadata") or {}
            if bool(pipeline_metadata.get("ocr_success")):
                ocr_success_true += 1
            doc_id = str(obj.get("doc_id") or "")
            if doc_id in target_doc_ids:
                updated_rows_seen += 1
                clean_text = (clean_dir / f"{doc_id}.md").read_text(encoding="utf-8").rstrip("\n")
                if hashlib.sha256(clean_text.encode("utf-8")).hexdigest() == hashlib.sha256(str(obj.get("text") or "").encode("utf-8")).hexdigest():
                    matched_text_sha_ok += 1
    summary = {
        "total_rows": int(total_rows),
        "ocr_success_true": int(ocr_success_true),
        "updated_rows_seen": int(updated_rows_seen),
        "matched_text_sha_ok": int(matched_text_sha_ok),
        "expected_updated_rows": int(len(target_doc_ids)),
    }
    (run_root / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if updated_rows_seen != len(target_doc_ids):
        raise RuntimeError(f"updated row count mismatch: seen={updated_rows_seen} expected={len(target_doc_ids)}")
    if matched_text_sha_ok != updated_rows_seen:
        raise RuntimeError(f"text hash mismatch on {updated_rows_seen - matched_text_sha_ok} updated rows")
    LOGGER.info("Validated staged OpenArchives repo for %d cleaned rows", updated_rows_seen)
    return summary


def upload_staged_repo(staged_root: Path, repo_id: str, revision: Optional[str], num_workers: int) -> None:
    token = get_token()
    if not token:
        raise RuntimeError("HF token is not available")
    api = HfApi(token=token)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=staged_root,
        revision=revision,
        ignore_patterns=[".cache/**"],
        num_workers=num_workers,
        print_report=True,
        print_report_every=60,
    )


def run_end_to_end(args: argparse.Namespace) -> None:
    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    hf_root = args.hf_root
    if not args.skip_download:
        LOGGER.info("Downloading %s into %s", args.repo_id, hf_root)
        _snapshot_download_hf_repo(args.repo_id, hf_root, args.revision)
    target_df = build_target_manifest(hf_root, args.ocr_manifest, run_root, merged_markdown_root=args.merged_markdown_root)
    paths = run_fast_clean_debug(
        target_df,
        run_root,
        render_workers=int(args.render_workers),
        min_progress_steps=int(args.min_progress_steps),
        min_repeat_steps=int(args.min_repeat_steps),
        min_same_digit_steps=int(args.min_same_digit_steps),
        word_rep_threshold=int(args.word_rep_threshold),
        word_min_period=int(args.word_min_period),
        word_window=int(args.word_window),
    )
    metrics_df = reevaluate_cleaned_outputs(
        target_df,
        paths["clean_dir"],
        run_root,
        num_threads=int(args.num_threads),
        min_repeat_run=int(args.min_repeat_run),
    )
    patch_summary = patch_openarchives_dataset(hf_root, target_df, metrics_df, paths["clean_dir"], run_root)
    validation = validate_patched_dataset(Path(patch_summary["staged_root"]), target_df, paths["clean_dir"], run_root)
    final_report = {
        "run_root": str(run_root),
        "target_manifest": str(run_root / "target_manifest.parquet"),
        "clean_dir": str(paths["clean_dir"]),
        "debug_dir": str(paths["debug_dir"]),
        "clean_metrics": str(run_root / "clean_metrics.parquet"),
        "patch_summary": patch_summary,
        "validation": validation,
    }
    if args.upload:
        upload_staged_repo(Path(patch_summary["staged_root"]), args.repo_id, args.revision, int(args.upload_workers))
        final_report["uploaded"] = True
    else:
        final_report["uploaded"] = False
    (run_root / "final_report.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(getattr(args, "verbose", False))

    if args.command == "build-target-manifest":
        build_target_manifest(args.hf_root, args.ocr_manifest, args.run_root, merged_markdown_root=args.merged_markdown_root)
        return
    if args.command == "run-clean":
        target_df = _load_target_manifest(args.target_manifest)
        run_fast_clean_debug(
            target_df,
            args.run_root,
            render_workers=int(args.render_workers),
            min_progress_steps=int(args.min_progress_steps),
            min_repeat_steps=int(args.min_repeat_steps),
            min_same_digit_steps=int(args.min_same_digit_steps),
            word_rep_threshold=int(args.word_rep_threshold),
            word_min_period=int(args.word_min_period),
            word_window=int(args.word_window),
        )
        return
    if args.command == "reevaluate-cleaned":
        target_df = _load_target_manifest(args.target_manifest)
        reevaluate_cleaned_outputs(target_df, args.clean_dir, args.run_root, num_threads=int(args.num_threads), min_repeat_run=int(args.min_repeat_run))
        return
    if args.command == "patch-dataset":
        target_df = _load_target_manifest(args.target_manifest)
        metrics_df = pd.read_parquet(args.metrics_parquet)
        patch_openarchives_dataset(args.hf_root, target_df, metrics_df, args.clean_dir, args.run_root)
        return
    if args.command == "validate-patch":
        target_df = _load_target_manifest(args.target_manifest)
        validate_patched_dataset(args.staged_root, target_df, args.clean_dir, args.run_root)
        return
    if args.command == "upload":
        upload_staged_repo(args.staged_root, args.repo_id, args.revision, int(args.num_workers))
        return
    if args.command == "run-end-to-end":
        run_end_to_end(args)
        return
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
