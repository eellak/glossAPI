from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from glossapi.ocr.deepseek.scheduling import (
    SourceDocument,
    assign_batches_to_lanes,
    build_exact_fill_batches,
    build_fixed_shard_slices,
    build_whole_document_slices,
    pack_slices_into_batches,
)


def _parse_devices(spec: str) -> List[int]:
    tokens = [piece.strip() for piece in str(spec or "").split(",") if piece.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("--devices must contain at least one GPU id")
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid GPU list: {spec}") from exc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.deepseek_pipeline_benchmark",
        description="Benchmark DeepSeek OCR pipeline throughput for different scheduling strategies.",
    )
    p.add_argument("--repo", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--python-bin", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--mode", default="static", choices=["static", "streaming"])
    p.add_argument(
        "--scheduler",
        default="whole_doc",
        choices=["whole_doc", "fixed_shard", "exact_fill"],
    )
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--doc-order", default="name", choices=["name", "random", "largest_first"])
    p.add_argument("--seed", type=int, default=20260330)
    p.add_argument("--target-batch-pages", type=int, default=160)
    p.add_argument("--stream-batch-pages", type=int, default=160)
    p.add_argument("--shard-pages", type=int, default=0)
    p.add_argument("--shard-threshold-pages", type=int, default=0)
    p.add_argument("--runtime-backend", default="vllm", choices=["transformers", "vllm"])
    p.add_argument("--ocr-profile", default="markdown_grounded", choices=["markdown_grounded", "plain_ocr"])
    p.add_argument("--prompt-override", default=None)
    p.add_argument("--repair-mode", default="auto", choices=["auto", "off"])
    p.add_argument("--attn-backend", default="auto")
    p.add_argument("--base-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--render-dpi", type=int, default=144)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--vllm-batch-size", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--disable-fp8-kv", action="store_true")
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def _weighted_documents(
    *,
    input_dir: Path,
    max_docs: Optional[int],
    doc_order: str,
    seed: int,
) -> List[SourceDocument]:
    from glossapi.ocr.deepseek import runner as deepseek_runner

    documents = [
        SourceDocument(name=path.name, pages=int(deepseek_runner._effective_page_count(path, None)))
        for path in sorted(input_dir.glob("*.pdf"))
    ]
    if doc_order == "largest_first":
        documents.sort(key=lambda item: (-int(item.pages), str(item.name)))
    elif doc_order == "random":
        rng = random.Random(int(seed))
        rng.shuffle(documents)
    if max_docs is not None:
        documents = documents[: max(0, int(max_docs))]
    return documents


def _plan_lanes(
    *,
    documents: List[SourceDocument],
    devices: List[int],
    workers_per_gpu: int,
    scheduler: str,
    target_batch_pages: int,
    shard_pages: int,
    shard_threshold_pages: int,
) -> List[Dict[str, Any]]:
    scheduler_norm = str(scheduler or "whole_doc").strip().lower()
    if scheduler_norm == "exact_fill":
        batches = build_exact_fill_batches(documents, target_batch_pages=max(1, int(target_batch_pages)))
    else:
        if scheduler_norm == "fixed_shard":
            slices = build_fixed_shard_slices(
                documents,
                shard_pages=max(1, int(shard_pages)),
                shard_threshold_pages=max(0, int(shard_threshold_pages)),
            )
        else:
            slices = build_whole_document_slices(documents)
        batches = pack_slices_into_batches(slices, target_batch_pages=max(1, int(target_batch_pages)))
    lanes = assign_batches_to_lanes(
        batches,
        devices=devices,
        workers_per_gpu=max(1, int(workers_per_gpu)),
    )
    return [lane.to_dict() for lane in lanes if lane.batches]


def _collect_repair_metrics(run_dir: Path) -> Dict[str, int]:
    metrics_dir = run_dir / "json" / "metrics"
    totals = {
        "docs_with_metrics": 0,
        "pages_flagged": 0,
        "pages_repaired": 0,
        "plain_repairs": 0,
        "tiled_repairs": 0,
    }
    if not metrics_dir.exists():
        return totals
    for path in metrics_dir.glob("*.metrics.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        totals["docs_with_metrics"] += 1
        summary = data.get("repair_summary") or {}
        totals["pages_flagged"] += int(summary.get("pages_flagged", 0))
        totals["pages_repaired"] += int(summary.get("pages_repaired", 0))
        totals["plain_repairs"] += int(summary.get("plain_repairs", 0))
        totals["tiled_repairs"] += int(summary.get("tiled_repairs", 0))
    return totals


def _collect_runtime_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "sidecars" / "ocr_runtime" / "runtime_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _flatten_lane_batches(lane: Dict[str, Any]) -> Dict[str, Any]:
    files: List[str] = []
    page_ranges: List[str] = []
    pages = 0
    planned_batch_pages: List[int] = []
    for batch in list(lane.get("batches") or []):
        batch_pages = int(batch.get("pages", 0))
        pages += batch_pages
        planned_batch_pages.append(batch_pages)
        files.extend(list(batch.get("files") or []))
        page_ranges.extend(list(batch.get("page_ranges") or []))
    return {
        "files": files,
        "page_ranges": page_ranges,
        "pages": int(pages),
        "planned_batch_count": len(planned_batch_pages),
        "planned_batch_pages": planned_batch_pages,
    }


def main() -> int:
    args = _parse_args()
    repo = Path(args.repo).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    python_bin = Path(args.python_bin).expanduser()
    model_dir = Path(args.model_dir).resolve()
    devices = _parse_devices(args.devices)

    from glossapi.ocr.deepseek import runner as deepseek_runner

    documents = _weighted_documents(
        input_dir=input_dir,
        max_docs=args.max_docs,
        doc_order=args.doc_order,
        seed=int(args.seed),
    )
    if not documents:
        raise SystemExit("No PDFs found for benchmark input set.")
    lanes = _plan_lanes(
        documents=documents,
        devices=devices,
        workers_per_gpu=max(1, int(args.workers_per_gpu)),
        scheduler=str(args.scheduler),
        target_batch_pages=int(args.target_batch_pages),
        shard_pages=int(args.shard_pages),
        shard_threshold_pages=int(args.shard_threshold_pages),
    )

    run_dir = output_root / args.label
    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "lane_plan.json").write_text(json.dumps(lanes, indent=2), encoding="utf-8")

    script_path = (
        deepseek_runner.DEFAULT_VLLM_SCRIPT
        if str(args.runtime_backend) == "vllm"
        else deepseek_runner.DEFAULT_SCRIPT
    )
    py_env = {"PYTHONPATH": str(repo / "src")}

    def start_lane(lane: Dict[str, Any]) -> Dict[str, Any]:
        lane_id = int(lane["lane_id"])
        visible_device = int(lane["visible_device"])
        lane_plan = _flatten_lane_batches(lane)
        files = list(lane_plan["files"])
        page_ranges = list(lane_plan["page_ranges"])
        pages = int(lane_plan["pages"])
        resolved_vllm_batch_size = (
            int(args.vllm_batch_size)
            if args.vllm_batch_size is not None
            else min(max(1, int(args.target_batch_pages)), max(1, pages))
        )
        log_path = logs_dir / f"lane_{lane_id:02d}_gpu{visible_device}.log"
        fh = log_path.open("w", encoding="utf-8")
        cmd = deepseek_runner._build_cli_command(
            input_dir=input_dir,
            output_dir=run_dir,
            files=files,
            page_ranges=page_ranges,
            model_dir=model_dir,
            python_bin=python_bin,
            script=script_path,
            max_pages=None,
            content_debug=False,
            device="cuda",
            ocr_profile=str(args.ocr_profile),
            prompt_override=args.prompt_override,
            attn_backend=str(args.attn_backend),
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=None,
            render_dpi=int(args.render_dpi),
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=None,
            no_repeat_ngram_size=None,
            runtime_backend=str(args.runtime_backend),
            vllm_batch_size=resolved_vllm_batch_size,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            disable_fp8_kv=bool(args.disable_fp8_kv),
            repair_mode=str(args.repair_mode),
        )
        env = deepseek_runner._build_env(python_bin=python_bin, visible_device=visible_device)
        env["PYTHONPATH"] = f"{py_env['PYTHONPATH']}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else py_env["PYTHONPATH"]
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)  # nosec: controlled args
        return {
            "lane_id": lane_id,
            "visible_device": visible_device,
            "batch_id": 0,
            "pages": pages,
            "files": files,
            "page_ranges": page_ranges,
            "planned_batch_count": int(lane_plan["planned_batch_count"]),
            "planned_batch_pages": list(lane_plan["planned_batch_pages"]),
            "resolved_vllm_batch_size": resolved_vllm_batch_size,
            "log_path": str(log_path),
            "fh": fh,
            "proc": proc,
            "start_ts": time.perf_counter(),
            "cmd": cmd,
        }

    global_start = time.perf_counter()
    active: List[Dict[str, Any]] = [start_lane(lane) for lane in lanes]

    batch_results: List[Dict[str, Any]] = []
    while active:
        time.sleep(0.2)
        for item in list(active):
            rc = item["proc"].poll()
            if rc is None:
                continue
            end_ts = time.perf_counter()
            item["fh"].close()
            elapsed = max(0.000001, float(end_ts - item["start_ts"]))
            batch_results.append(
                {
                    "lane_id": int(item["lane_id"]),
                    "visible_device": int(item["visible_device"]),
                    "batch_id": int(item["batch_id"]),
                    "pages": int(item["pages"]),
                    "files": list(item["files"]),
                    "page_ranges": list(item.get("page_ranges") or []),
                    "planned_batch_count": int(item.get("planned_batch_count", 1)),
                    "planned_batch_pages": list(item.get("planned_batch_pages") or []),
                    "return_code": int(rc),
                    "resolved_vllm_batch_size": int(item["resolved_vllm_batch_size"]),
                    "start_offset_sec": float(item["start_ts"] - global_start),
                    "end_offset_sec": float(end_ts - global_start),
                    "elapsed_sec": float(elapsed),
                    "sec_per_page": float(elapsed / max(1, int(item["pages"]))),
                    "log_path": str(item["log_path"]),
                    "cmd": item["cmd"],
                }
            )
            active.remove(item)

    total_elapsed = max(0.000001, time.perf_counter() - global_start)
    total_pages = sum(int(doc.pages) for doc in documents)
    failures = [item for item in batch_results if int(item["return_code"]) != 0]

    lane_results: List[Dict[str, Any]] = []
    for lane in lanes:
        lane_batches = [item for item in batch_results if int(item["lane_id"]) == int(lane["lane_id"])]
        if not lane_batches:
            continue
        lane_start = min(float(item["start_offset_sec"]) for item in lane_batches)
        lane_end = max(float(item["end_offset_sec"]) for item in lane_batches)
        lane_elapsed = max(0.000001, lane_end - lane_start)
        lane_pages = sum(int(item["pages"]) for item in lane_batches)
        lane_results.append(
            {
                "lane_id": int(lane["lane_id"]),
                "visible_device": int(lane["visible_device"]),
                "batch_count": len(lane_batches),
                "pages": int(lane_pages),
                "active_elapsed_sec": float(lane_elapsed),
                "sec_per_page": float(lane_elapsed / max(1, lane_pages)),
                "all_return_codes_zero": all(int(item["return_code"]) == 0 for item in lane_batches),
            }
        )

    gpu_results: List[Dict[str, Any]] = []
    for visible_device in sorted({int(item["visible_device"]) for item in batch_results}):
        gpu_batches = [item for item in batch_results if int(item["visible_device"]) == visible_device]
        gpu_start = min(float(item["start_offset_sec"]) for item in gpu_batches)
        gpu_end = max(float(item["end_offset_sec"]) for item in gpu_batches)
        gpu_elapsed = max(0.000001, gpu_end - gpu_start)
        gpu_pages = sum(int(item["pages"]) for item in gpu_batches)
        gpu_results.append(
            {
                "visible_device": visible_device,
                "batch_count": len(gpu_batches),
                "pages": int(gpu_pages),
                "active_elapsed_sec": float(gpu_elapsed),
                "sec_per_page": float(gpu_elapsed / max(1, gpu_pages)),
                "all_return_codes_zero": all(int(item["return_code"]) == 0 for item in gpu_batches),
            }
        )

    repair_metrics = _collect_repair_metrics(run_dir)
    runtime_summary = _collect_runtime_summary(run_dir)
    summary = {
        "label": str(args.label),
        "status": "pass" if not failures else "fail",
        "mode": str(args.mode),
        "scheduler": str(args.scheduler),
        "runtime_backend": str(args.runtime_backend),
        "ocr_profile": str(args.ocr_profile),
        "repair_mode": str(args.repair_mode),
        "devices": devices,
        "workers_per_gpu": int(args.workers_per_gpu),
        "doc_order": str(args.doc_order),
        "target_batch_pages": int(args.target_batch_pages),
        "stream_batch_pages": int(args.stream_batch_pages),
        "docs": len(documents),
        "pages": int(total_pages),
        "shard_pages": int(args.shard_pages),
        "shard_threshold_pages": int(args.shard_threshold_pages),
        "wall_time_sec": float(total_elapsed),
        "sec_per_page": float(total_elapsed / max(1, total_pages)),
        "batch_results": batch_results,
        "lane_results": lane_results,
        "gpu_results": gpu_results,
        "repair_metrics": repair_metrics,
        "runtime_summary": runtime_summary,
        "steady_state": dict(runtime_summary.get("steady_state") or {}),
        "failures": failures,
    }
    (run_dir / "pipeline_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
