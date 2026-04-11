"""Math runtime orchestration helpers."""

from __future__ import annotations

import os
import queue
import time
from pathlib import Path
from typing import Any, Dict, List

from .context import CorpusOcrContext
from ..corpus_utils import _maybe_import_torch
from .config import OcrRequest
from .math_worker import gpu_math_worker


def run_math_enrichment(
    context: CorpusOcrContext,
    *,
    stems: List[str],
    request: OcrRequest,
    skip_mgr,
    skiplist_path: Path,
) -> None:
    if not stems:
        context.logger.info("No Docling JSON found for math enrichment.")
        return

    initial_math_targets = len(stems)
    current_skips = skip_mgr.reload() if skip_mgr else set()
    if current_skips:
        before = len(stems)
        stems = [stem for stem in stems if stem not in current_skips]
        removed = before - len(stems)
        if removed:
            context.logger.warning(
                "Skip-list %s filtered %d document(s) from Phase-2 math.",
                skiplist_path,
                removed,
            )
        if not stems:
            context.logger.info("All math targets filtered by skip-list; nothing to do.")
            return

    context.logger.info(
        "Math targets: total=%d kept=%d filtered_skiplist=%d",
        initial_math_targets,
        len(stems),
        initial_math_targets - len(stems),
    )

    local_targets = None
    if request.math_targets:
        local_targets = {stem: request.math_targets.get(stem) for stem in stems if stem in request.math_targets}

    if request.use_gpus.lower() != "multi":
        context.formula_enrich_from_json(
            files=stems,
            device=(request.device or "cuda"),
            batch_size=int(request.math_batch_size),
            dpi_base=int(request.math_dpi_base),
            targets_by_stem=local_targets,
        )
        return

    devs = list(request.devices or [])
    if not devs:
        try:
            import subprocess

            proc = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0 and proc.stdout:
                for line in proc.stdout.splitlines():
                    if line.startswith("GPU "):
                        try:
                            devs.append(int(line.split(":", 1)[0].split()[1]))
                        except Exception:
                            pass
        except Exception:
            pass
        if not devs:
            torch_mod = _maybe_import_torch()
            try:
                if torch_mod is not None and getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available():
                    devs = list(range(torch_mod.cuda.device_count()))
            except Exception:
                pass

    if not devs:
        msg = "Multi-GPU math requested but no GPUs detected; aborting math enhancement"
        context.logger.error(msg)
        raise RuntimeError(msg)

    from multiprocessing import get_context

    ctx = get_context("spawn")
    work_q = ctx.Queue()
    result_q = ctx.Queue()
    manager = ctx.Manager()
    status_map = manager.dict()

    for stem in stems:
        work_q.put(stem)

    worker_log_dir_env = os.environ.get("GLOSSAPI_WORKER_LOG_DIR")
    worker_log_dir_to_use = worker_log_dir_env
    if not worker_log_dir_to_use:
        default_worker_log_dir = context.logs_dir / "math_workers"
        try:
            default_worker_log_dir.mkdir(parents=True, exist_ok=True)
            worker_log_dir_to_use = str(default_worker_log_dir)
        except Exception as exc:
            context.logger.warning(
                "Unable to prepare worker log directory %s: %s",
                default_worker_log_dir,
                exc,
            )
            worker_log_dir_to_use = None
    if worker_log_dir_to_use:
        os.environ["GLOSSAPI_WORKER_LOG_DIR"] = worker_log_dir_to_use
    marker_base = Path(worker_log_dir_to_use) if worker_log_dir_to_use else (context.logs_dir / "math_workers")
    try:
        marker_base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    marker_files = {dev_id: marker_base / f"gpu{dev_id}.current" for dev_id in devs}

    procs: List[Any] = []
    active: List[Any] = []
    proc_gpu: Dict[int, int] = {}
    try:
        respawn_cap = int(os.environ.get("GLOSSAPI_MATH_RESPAWN_CAP", "5"))
    except Exception:
        respawn_cap = 5
    respawn_cap = max(0, respawn_cap)
    respawn_counts: Dict[int, int] = {dev_id: 0 for dev_id in devs}

    for dev_id in devs:
        proc = ctx.Process(
            target=gpu_math_worker,
            args=(
                dev_id,
                str(context.input_dir),
                str(context.output_dir),
                work_q,
                int(request.math_batch_size),
                int(request.math_dpi_base),
                request.device or "cuda",
                local_targets or {},
                result_q,
                status_map,
                str(marker_base),
            ),
        )
        proc.start()
        procs.append(proc)
        active.append(proc)
        if proc.pid is not None:
            proc_gpu[proc.pid] = dev_id

    try:
        last_summary = time.time()
        while active:
            for proc in list(active):
                proc.join(timeout=0.05)
                if proc.is_alive():
                    continue
                active.remove(proc)
                if proc in procs:
                    procs.remove(proc)
                pid = proc.pid or -1
                gpu_id = proc_gpu.pop(pid, None)
                exitcode = proc.exitcode
                stems_for_skip: List[str] = []
                if gpu_id is not None:
                    current_entry = status_map.pop(gpu_id, None)
                    if current_entry:
                        if isinstance(current_entry, (list, tuple, set)):
                            entries = list(current_entry)
                        else:
                            entries = [current_entry]
                        stems_for_skip = [str(item) for item in entries if item]
                    marker_path = marker_files.get(gpu_id)
                    if marker_path:
                        try:
                            marker_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                if exitcode not in (0, None) and gpu_id is not None:
                    if stems_for_skip:
                        skip_mgr.add(stems_for_skip)
                    context.logger.warning("Math worker GPU%s exited with %s", gpu_id, exitcode)
                    respawn_counts[gpu_id] = respawn_counts.get(gpu_id, 0) + 1
                    attempts = respawn_counts[gpu_id]
                    if respawn_cap and attempts > respawn_cap:
                        context.logger.error(
                            "Math worker GPU%s exceeded respawn cap (%s); not respawning",
                            gpu_id,
                            respawn_cap,
                        )
                        continue
                    replacement = ctx.Process(
                        target=gpu_math_worker,
                        args=(
                            gpu_id,
                            str(context.input_dir),
                            str(context.output_dir),
                            work_q,
                            int(request.math_batch_size),
                            int(request.math_dpi_base),
                            request.device or "cuda",
                            local_targets or {},
                            result_q,
                            status_map,
                            str(marker_base),
                        ),
                    )
                    replacement.start()
                    procs.append(replacement)
                    active.append(replacement)
                    if replacement.pid is not None:
                        proc_gpu[replacement.pid] = gpu_id
                    continue

            while True:
                try:
                    event = result_q.get_nowait()
                except queue.Empty:
                    break
                if not event:
                    continue
                if event.get("event") == "math_batch":
                    stems_bad = event.get("problematic", [])
                    if stems_bad:
                        skip_mgr.add(stems_bad)
                    worker = event.get("worker")
                    try:
                        worker_gpu = int(worker)
                    except Exception:
                        worker_gpu = None
                    if worker_gpu is not None:
                        status_map.pop(worker_gpu, None)
                        marker_path = marker_files.get(worker_gpu)
                        if marker_path:
                            try:
                                marker_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                elif event.get("event") == "exit" and event.get("exitcode", 0) not in (0, None):
                    context.logger.warning(
                        "Math worker GPU%s reported exit code %s",
                        event.get("worker"),
                        event.get("exitcode"),
                    )

            now = time.time()
            if now - last_summary > 30:
                try:
                    qsize = work_q.qsize()
                except NotImplementedError:
                    qsize = -1
                context.logger.info(
                    "Math progress: queue=%d active_workers=%d",
                    qsize,
                    len(active),
                )
                last_summary = now

            if not active:
                break

        remaining_after_cap: List[str] = []
        try:
            while True:
                item = work_q.get_nowait()
                if isinstance(item, str) and item.strip():
                    remaining_after_cap.append(item)
        except queue.Empty:
            pass
        if remaining_after_cap:
            skip_mgr.add(remaining_after_cap)
            context.logger.error(
                "No active math workers remain; skipped %d pending item(s)",
                len(remaining_after_cap),
            )
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.join()
        try:
            manager.shutdown()
        except Exception:
            pass
        if worker_log_dir_env is not None:
            os.environ["GLOSSAPI_WORKER_LOG_DIR"] = worker_log_dir_env
        else:
            os.environ.pop("GLOSSAPI_WORKER_LOG_DIR", None)
