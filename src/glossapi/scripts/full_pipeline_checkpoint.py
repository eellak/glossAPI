from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from glossapi import Corpus
from glossapi.scripts.extract_checkpoint_benchmark import _apply_cli_tuning_overrides


def _parse_int_list(values: Optional[List[int]]) -> List[int]:
    return list(values or [])


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.full_pipeline_checkpoint",
        description=(
            "Run a sample GlossAPI pipeline checkpoint from extract through JSONL export "
            "and write a compact timing/continuity report."
        ),
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--export-path", required=True)
    p.add_argument("--report-path", required=True)
    p.add_argument("--clean-output-dir", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-clean", action="store_true")
    p.add_argument("--skip-ocr", action="store_true")

    p.add_argument("--phase1-backend", default="docling", choices=["auto", "safe", "docling"])
    p.add_argument("--accel-type", default="CUDA")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--use-gpus", default="single", choices=["single", "multi"])
    p.add_argument("--devices", nargs="*", type=int, default=None)
    p.add_argument("--workers-per-device", type=int, default=1)
    p.add_argument("--benchmark-mode", action="store_true")
    p.add_argument("--filenames", nargs="*", default=[])
    p.add_argument("--drop-bad", action="store_true")

    p.add_argument("--docling-max-batch-files", type=int, default=None)
    p.add_argument("--docling-batch-target-pages", type=int, default=None)
    p.add_argument("--docling-layout-batch-size", type=int, default=None)
    p.add_argument("--docling-table-batch-size", type=int, default=None)
    p.add_argument("--docling-ocr-batch-size", type=int, default=None)
    p.add_argument("--docling-page-batch-size", type=int, default=None)

    p.add_argument("--ocr-backend", default="deepseek")
    p.add_argument("--ocr-runtime-backend", default="vllm")
    p.add_argument("--ocr-use-gpus", default="single", choices=["single", "multi"])
    p.add_argument("--ocr-devices", nargs="*", type=int, default=None)
    p.add_argument("--ocr-workers-per-gpu", type=int, default=1)
    p.add_argument("--ocr-vllm-batch-size", type=int, default=None)
    p.add_argument("--ocr-repair-exec-batch-target-pages", type=int, default=None)
    p.add_argument("--ocr-repair-exec-batch-target-items", type=int, default=None)
    p.add_argument("--ocr-target-batch-pages", type=int, default=160)
    p.add_argument("--ocr-render-dpi", type=int, default=None)
    p.add_argument("--ocr-scheduler", default="auto")
    p.add_argument("--ocr-math-enhance", action="store_true")

    p.add_argument("--text-key", default="text")
    p.add_argument("--metadata-key", default="pipeline_metadata")
    return p.parse_args(argv)


def _read_metadata_counts(parquet_path: Path) -> Dict[str, int]:
    if not parquet_path.exists():
        return {
            "rows_total": 0,
            "needs_ocr_true": 0,
            "ocr_success_true": 0,
            "text_nonempty": 0,
        }
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return {
            "rows_total": 0,
            "needs_ocr_true": 0,
            "ocr_success_true": 0,
            "text_nonempty": 0,
        }
    text_series = df["text"] if "text" in df.columns else pd.Series([], dtype=object)
    text_nonempty = int(
        sum(bool(str(value).strip()) for value in text_series.fillna("").tolist())
    ) if len(text_series) else 0
    needs_ocr_true = int(df["needs_ocr"].fillna(False).astype(bool).sum()) if "needs_ocr" in df.columns else 0
    ocr_success_true = int(df["ocr_success"].fillna(False).astype(bool).sum()) if "ocr_success" in df.columns else 0
    return {
        "rows_total": int(len(df)),
        "needs_ocr_true": needs_ocr_true,
        "ocr_success_true": ocr_success_true,
        "text_nonempty": text_nonempty,
    }


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fp:
        return sum(1 for line in fp if line.strip())


def _export_jsonl_with_retry(
    corpus: Corpus,
    *,
    export_path: Path,
    metadata_path: Path,
    text_key: str,
    metadata_key: str,
    post_ocr_counts: Dict[str, int],
    max_attempts: int = 4,
    retry_delay_sec: float = 1.0,
) -> int:
    needs_retry = int(post_ocr_counts.get("text_nonempty", 0) or 0) > 0
    attempts = max_attempts if needs_retry else 1

    for attempt in range(attempts):
        if export_path.exists():
            export_path.unlink()
        corpus.jsonl(
            export_path,
            text_key=text_key,
            metadata_key=metadata_key,
            include_remaining_metadata=False,
            metadata_path=metadata_path,
        )
        export_records = _count_jsonl_records(export_path)
        if export_records > 0 or attempt == attempts - 1:
            return export_records
        time.sleep(retry_delay_sec)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _apply_cli_tuning_overrides(args)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    export_path = Path(args.export_path).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()

    if bool(args.clean_output_dir) and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = Corpus(input_dir=input_dir, output_dir=output_dir)
    metadata_path = output_dir / "download_results" / "download_results.parquet"

    started_at = time.time()
    skipped_phases: List[str] = []

    if bool(args.skip_extract):
        skipped_phases.append("extract")
        extract_elapsed = 0.0
    else:
        extract_start = time.perf_counter()
        corpus.extract(
            input_format="pdf",
            accel_type=str(args.accel_type),
            num_threads=int(args.num_threads),
            phase1_backend=str(args.phase1_backend),
            use_gpus=str(args.use_gpus),
            devices=_parse_int_list(args.devices),
            workers_per_device=int(args.workers_per_device),
            benchmark_mode=bool(args.benchmark_mode),
            filenames=list(args.filenames or []),
        )
        extract_elapsed = float(time.perf_counter() - extract_start)
    post_extract_counts = _read_metadata_counts(metadata_path)

    if bool(args.skip_clean):
        skipped_phases.append("clean")
        clean_elapsed = 0.0
    else:
        clean_start = time.perf_counter()
        corpus.clean(drop_bad=bool(args.drop_bad))
        clean_elapsed = float(time.perf_counter() - clean_start)
    post_clean_counts = _read_metadata_counts(metadata_path)

    if bool(args.skip_ocr):
        skipped_phases.append("ocr")
        ocr_elapsed = 0.0
    else:
        ocr_start = time.perf_counter()
        corpus.ocr(
            backend=str(args.ocr_backend),
            runtime_backend=str(args.ocr_runtime_backend),
            use_gpus=str(args.ocr_use_gpus),
            devices=_parse_int_list(args.ocr_devices),
            workers_per_gpu=int(args.ocr_workers_per_gpu),
            vllm_batch_size=args.ocr_vllm_batch_size,
            repair_exec_batch_target_pages=args.ocr_repair_exec_batch_target_pages,
            repair_exec_batch_target_items=args.ocr_repair_exec_batch_target_items,
            target_batch_pages=int(args.ocr_target_batch_pages),
            render_dpi=args.ocr_render_dpi,
            scheduler=str(args.ocr_scheduler),
            math_enhance=bool(args.ocr_math_enhance),
        )
        ocr_elapsed = float(time.perf_counter() - ocr_start)
    post_ocr_counts = _read_metadata_counts(metadata_path)

    export_start = time.perf_counter()
    export_records = _export_jsonl_with_retry(
        corpus,
        export_path=export_path,
        metadata_path=metadata_path,
        text_key=str(args.text_key),
        metadata_key=str(args.metadata_key),
        post_ocr_counts=post_ocr_counts,
    )
    export_elapsed = float(time.perf_counter() - export_start)

    finished_at = time.time()
    report: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "export_path": str(export_path),
        "metadata_path": str(metadata_path),
        "started_at": int(started_at),
        "finished_at": int(finished_at),
        "elapsed_total_sec": float(finished_at - started_at),
        "skipped_phases": list(skipped_phases),
        "extract_elapsed_sec": extract_elapsed,
        "clean_elapsed_sec": clean_elapsed,
        "ocr_elapsed_sec": ocr_elapsed,
        "export_elapsed_sec": export_elapsed,
        "post_extract_counts": post_extract_counts,
        "post_clean_counts": post_clean_counts,
        "post_ocr_counts": post_ocr_counts,
        "export_records": int(export_records),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "extract_elapsed_sec": round(extract_elapsed, 3),
                "clean_elapsed_sec": round(clean_elapsed, 3),
                "ocr_elapsed_sec": round(ocr_elapsed, 3),
                "export_elapsed_sec": round(export_elapsed, 3),
                "rows_total": post_ocr_counts["rows_total"],
                "needs_ocr_after_clean": post_clean_counts["needs_ocr_true"],
                "ocr_success_after_ocr": post_ocr_counts["ocr_success_true"],
                "export_records": int(export_records),
                "report_path": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
