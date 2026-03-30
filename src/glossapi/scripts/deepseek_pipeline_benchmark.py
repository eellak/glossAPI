from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        description="Benchmark DeepSeek OCR pipeline throughput for static and streaming-style scheduling.",
    )
    p.add_argument("--repo", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--python-bin", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--mode", default="static", choices=["static", "streaming"])
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--doc-order", default="name", choices=["name", "random", "largest_first"])
    p.add_argument("--seed", type=int, default=20260330)
    p.add_argument("--stream-batch-pages", type=int, default=160)
    p.add_argument("--runtime-backend", default="vllm", choices=["transformers", "vllm"])
    p.add_argument("--ocr-profile", default="markdown_grounded", choices=["markdown_grounded", "plain_ocr"])
    p.add_argument("--prompt-override", default=None)
    p.add_argument("--repair-mode", default="auto", choices=["auto", "off"])
    p.add_argument("--attn-backend", default="auto")
    p.add_argument("--base-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--render-dpi", type=int, default=144)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--vllm-batch-size", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--disable-fp8-kv", action="store_true")
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def _weighted_files(
    *,
    input_dir: Path,
    max_docs: Optional[int],
    doc_order: str,
    seed: int,
) -> List[Dict[str, Any]]:
    from glossapi.ocr.deepseek import runner as deepseek_runner

    weighted = []
    for path in sorted(input_dir.glob("*.pdf")):
        pages = int(deepseek_runner._effective_page_count(path, None))
        weighted.append({"name": path.name, "pages": pages})
    if doc_order == "largest_first":
        weighted.sort(key=lambda item: (-int(item["pages"]), str(item["name"])))
    elif doc_order == "random":
        rng = random.Random(int(seed))
        rng.shuffle(weighted)
    if max_docs is not None:
        weighted = weighted[: max(0, int(max_docs))]
    return weighted


def _empty_lanes(devices: List[int], workers_per_gpu: int) -> List[Dict[str, Any]]:
    lanes: List[Dict[str, Any]] = []
    lane_id = 0
    for visible_device in devices:
        for _ in range(max(1, int(workers_per_gpu))):
            lanes.append(
                {
                    "lane_id": lane_id,
                    "visible_device": int(visible_device),
                    "batches": [],
                    "assigned_pages": 0,
                }
            )
            lane_id += 1
    return lanes


def _plan_static(
    weighted_files: List[Dict[str, Any]],
    devices: List[int],
    workers_per_gpu: int,
    input_dir: Path,
) -> List[Dict[str, Any]]:
    from glossapi.ocr.deepseek import runner as deepseek_runner

    lanes = deepseek_runner._plan_lanes(
        file_list=[str(item["name"]) for item in weighted_files],
        input_root=input_dir,
        lane_devices=devices,
        workers_per_gpu=max(1, int(workers_per_gpu)),
        max_pages=None,
    )
    weights = {str(item["name"]): int(item["pages"]) for item in weighted_files}
    planned: List[Dict[str, Any]] = []
    for lane in lanes:
        files = list(lane["files"])
        if not files:
            continue
        weight = sum(int(weights.get(name, 0)) for name in files)
        planned.append(
            {
                "lane_id": int(lane["lane_id"]),
                "visible_device": int(lane["visible_device"]),
                "assigned_pages": int(weight),
                "batches": [
                    {
                        "batch_id": 0,
                        "files": files,
                        "pages": int(weight),
                    }
                ],
            }
        )
    return planned


def _plan_streaming(
    weighted_files: List[Dict[str, Any]],
    devices: List[int],
    workers_per_gpu: int,
    stream_batch_pages: int,
) -> List[Dict[str, Any]]:
    lanes = _empty_lanes(devices, workers_per_gpu)
    batch_target = max(1, int(stream_batch_pages))
    current: Dict[int, Dict[str, Any]] = {
        int(lane["lane_id"]): {"files": [], "pages": 0}
        for lane in lanes
    }

    def flush(lane: Dict[str, Any]) -> None:
        lane_id = int(lane["lane_id"])
        state = current[lane_id]
        if not state["files"]:
            return
        lane["batches"].append(
            {
                "batch_id": len(lane["batches"]),
                "files": list(state["files"]),
                "pages": int(state["pages"]),
            }
        )
        state["files"] = []
        state["pages"] = 0

    for item in weighted_files:
        lane = min(lanes, key=lambda value: (int(value["assigned_pages"]) + int(current[int(value["lane_id"])]["pages"]), int(value["lane_id"])))
        lane_id = int(lane["lane_id"])
        current[lane_id]["files"].append(str(item["name"]))
        current[lane_id]["pages"] = int(current[lane_id]["pages"]) + int(item["pages"])
        lane["assigned_pages"] = int(lane["assigned_pages"]) + int(item["pages"])
        if int(current[lane_id]["pages"]) >= batch_target:
            flush(lane)

    for lane in lanes:
        flush(lane)
    return [lane for lane in lanes if lane["batches"]]


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


def main() -> int:
    args = _parse_args()
    repo = Path(args.repo).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    python_bin = Path(args.python_bin).expanduser()
    model_dir = Path(args.model_dir).resolve()
    devices = _parse_devices(args.devices)

    from glossapi.ocr.deepseek import runner as deepseek_runner

    weighted_files = _weighted_files(
        input_dir=input_dir,
        max_docs=args.max_docs,
        doc_order=args.doc_order,
        seed=int(args.seed),
    )
    if not weighted_files:
        raise SystemExit("No PDFs found for benchmark input set.")

    if str(args.mode) == "streaming":
        lanes = _plan_streaming(
            weighted_files=weighted_files,
            devices=devices,
            workers_per_gpu=max(1, int(args.workers_per_gpu)),
            stream_batch_pages=max(1, int(args.stream_batch_pages)),
        )
    else:
        lanes = _plan_static(
            weighted_files=weighted_files,
            devices=devices,
            workers_per_gpu=max(1, int(args.workers_per_gpu)),
            input_dir=input_dir,
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

    def start_batch(lane: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        lane_id = int(lane["lane_id"])
        visible_device = int(lane["visible_device"])
        batch_id = int(batch["batch_id"])
        files = list(batch["files"])
        pages = int(batch["pages"])
        resolved_vllm_batch_size = (
            int(args.vllm_batch_size)
            if args.vllm_batch_size is not None
            else deepseek_runner._auto_vllm_batch_size(
                runtime_backend=str(args.runtime_backend),
                file_list=files,
                input_root=input_dir,
                max_pages=None,
            )
        )
        log_path = logs_dir / f"lane_{lane_id:02d}_batch_{batch_id:03d}_gpu{visible_device}.log"
        fh = log_path.open("w", encoding="utf-8")
        cmd = deepseek_runner._build_cli_command(
            input_dir=input_dir,
            output_dir=run_dir,
            files=files,
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
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = f"{py_env['PYTHONPATH']}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = py_env["PYTHONPATH"]
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)  # nosec: controlled args
        return {
            "lane_id": lane_id,
            "visible_device": visible_device,
            "batch_id": batch_id,
            "pages": pages,
            "files": files,
            "resolved_vllm_batch_size": resolved_vllm_batch_size,
            "log_path": str(log_path),
            "fh": fh,
            "proc": proc,
            "start_ts": time.perf_counter(),
            "cmd": cmd,
        }

    pending_batches: Dict[int, List[Dict[str, Any]]] = {
        int(lane["lane_id"]): list(lane["batches"])
        for lane in lanes
    }
    active: List[Dict[str, Any]] = []
    global_start = time.perf_counter()
    for lane in lanes:
        lane_id = int(lane["lane_id"])
        if pending_batches[lane_id]:
            first_batch = pending_batches[lane_id].pop(0)
            active.append(start_batch(lane, first_batch))

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
                    "return_code": int(rc),
                    "resolved_vllm_batch_size": item["resolved_vllm_batch_size"],
                    "start_offset_sec": float(item["start_ts"] - global_start),
                    "end_offset_sec": float(end_ts - global_start),
                    "elapsed_sec": float(elapsed),
                    "sec_per_page": float(elapsed / max(1, int(item["pages"]))),
                    "log_path": str(item["log_path"]),
                    "cmd": item["cmd"],
                }
            )
            active.remove(item)
            lane = next(lane for lane in lanes if int(lane["lane_id"]) == int(item["lane_id"]))
            if pending_batches[int(item["lane_id"])]:
                next_batch = pending_batches[int(item["lane_id"])].pop(0)
                active.append(start_batch(lane, next_batch))

    total_elapsed = max(0.000001, time.perf_counter() - global_start)
    total_pages = sum(int(item["pages"]) for item in weighted_files)
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
    summary = {
        "label": str(args.label),
        "status": "pass" if not failures else "fail",
        "mode": str(args.mode),
        "runtime_backend": str(args.runtime_backend),
        "ocr_profile": str(args.ocr_profile),
        "repair_mode": str(args.repair_mode),
        "devices": devices,
        "workers_per_gpu": int(args.workers_per_gpu),
        "doc_order": str(args.doc_order),
        "stream_batch_pages": int(args.stream_batch_pages),
        "docs": len(weighted_files),
        "pages": int(total_pages),
        "wall_time_sec": float(total_elapsed),
        "sec_per_page": float(total_elapsed / max(1, total_pages)),
        "batch_results": batch_results,
        "lane_results": lane_results,
        "gpu_results": gpu_results,
        "repair_metrics": repair_metrics,
        "failures": failures,
    }
    (run_dir / "pipeline_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
