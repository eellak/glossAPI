"""DeepSeek OCR runner."""

from __future__ import annotations

from contextlib import ExitStack
import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from glossapi.ocr.deepseek.defaults import (
    DEFAULT_ATTN_BACKEND,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OCR_PROFILE,
    DEFAULT_RENDER_DPI,
    DEFAULT_REPAIR_MODE,
    DEFAULT_RUNTIME_BACKEND,
    DEFAULT_TARGET_BATCH_PAGES,
    DEFAULT_WORKERS_PER_GPU,
    resolve_gpu_memory_utilization,
    resolve_render_dpi,
)
from glossapi.ocr.deepseek.launcher import _build_cli_command, _build_env, _run_cli
from glossapi.ocr.deepseek import runner_planning as _planning
from glossapi.ocr.deepseek import runner_reassembly as _reassembly
from glossapi.ocr.deepseek import runner_runtime_support as _runtime_support
from glossapi.ocr.deepseek.runtime_paths import resolve_deepseek_python
from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _join_page_outputs, _split_page_outputs, _write_outputs
from glossapi.ocr.deepseek.work_queue import (
    STATUS_DONE,
    STATUS_FAILED,
    init_work_db,
    iter_work_items,
    requeue_worker_batches,
    work_queue_counts,
)

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCRIPT = REPO_ROOT / "src" / "glossapi" / "ocr" / "deepseek" / "run_pdf_ocr_transformers.py"
DEFAULT_VLLM_SCRIPT = REPO_ROOT / "src" / "glossapi" / "ocr" / "deepseek" / "run_pdf_ocr_vllm.py"
AUTO_VLLM_BATCH_PAGE_CAP = _planning.AUTO_VLLM_BATCH_PAGE_CAP
DEFAULT_WORKER_RESPAWN_CAP = 3
DEFAULT_WORK_ITEM_MAX_ATTEMPTS = 2
DEFAULT_WORK_STALE_AFTER_SEC = 900.0
DEFAULT_WORK_HEARTBEAT_SEC = 10.0
DEFAULT_TELEMETRY_INTERVAL_SEC = 15.0


def _page_count(pdf_path: Path) -> int:
    return _planning.page_count(pdf_path)


def _parse_device_index(device: Optional[str]) -> Optional[int]:
    return _planning.parse_device_index(device)


def _detect_visible_gpus() -> List[int]:
    return _planning.detect_visible_gpus()


def _resolve_lane_devices(
    *,
    use_gpus: Optional[str],
    devices: Optional[List[int]],
    workers_per_gpu: int,
    device: Optional[str],
) -> List[int]:
    return _planning.resolve_lane_devices(
        use_gpus=use_gpus,
        devices=devices,
        workers_per_gpu=workers_per_gpu,
        device=device,
        detect_visible_gpus_fn=_detect_visible_gpus,
        parse_device_index_fn=_parse_device_index,
    )


def _effective_page_count(pdf_path: Path, max_pages: Optional[int]) -> int:
    return _planning.effective_page_count(pdf_path, max_pages, page_count_fn=_page_count)


def _plan_lanes(
    *,
    file_list: List[str],
    input_root: Path,
    lane_devices: List[int],
    workers_per_gpu: int,
    max_pages: Optional[int],
) -> List[Dict[str, Any]]:
    return _planning.plan_lanes(
        file_list=file_list,
        input_root=input_root,
        lane_devices=lane_devices,
        workers_per_gpu=workers_per_gpu,
        max_pages=max_pages,
        page_count_fn=_page_count,
    )


def _resolve_scheduler(
    *,
    scheduler: Optional[str],
    runtime_backend: str,
    lane_devices: List[int],
    workers_per_gpu: int,
) -> str:
    return _planning.resolve_scheduler(
        scheduler=scheduler,
        runtime_backend=runtime_backend,
        lane_devices=lane_devices,
        workers_per_gpu=workers_per_gpu,
    )


def _plan_lane_batches(
    *,
    file_list: List[str],
    input_root: Path,
    lane_devices: List[int],
    workers_per_gpu: int,
    max_pages: Optional[int],
    runtime_backend: str,
    scheduler: Optional[str],
    target_batch_pages: int,
    shard_pages: int,
    shard_threshold_pages: int,
) -> List[Dict[str, Any]]:
    return _planning.plan_lane_batches(
        file_list=file_list,
        input_root=input_root,
        lane_devices=lane_devices,
        workers_per_gpu=workers_per_gpu,
        max_pages=max_pages,
        runtime_backend=runtime_backend,
        scheduler=scheduler,
        target_batch_pages=target_batch_pages,
        shard_pages=shard_pages,
        shard_threshold_pages=shard_threshold_pages,
        page_count_fn=_page_count,
    )


def _plan_work_batches(
    *,
    file_list: List[str],
    input_root: Path,
    max_pages: Optional[int],
    runtime_backend: str,
    scheduler: Optional[str],
    lane_devices: List[int],
    workers_per_gpu: int,
    target_batch_pages: int,
    shard_pages: int,
    shard_threshold_pages: int,
) -> List[Dict[str, Any]]:
    return _planning.plan_work_batches(
        file_list=file_list,
        input_root=input_root,
        max_pages=max_pages,
        runtime_backend=runtime_backend,
        scheduler=scheduler,
        lane_devices=lane_devices,
        workers_per_gpu=workers_per_gpu,
        target_batch_pages=target_batch_pages,
        shard_pages=shard_pages,
        shard_threshold_pages=shard_threshold_pages,
        page_count_fn=_page_count,
    )


def _auto_vllm_batch_size(
    *,
    runtime_backend: str,
    file_list: List[str],
    input_root: Path,
    max_pages: Optional[int],
) -> Optional[int]:
    return _planning.auto_vllm_batch_size(
        runtime_backend=runtime_backend,
        file_list=file_list,
        input_root=input_root,
        max_pages=max_pages,
        page_count_fn=_page_count,
    )


def _auto_vllm_batch_size_for_pages(*, runtime_backend: str, pages: int) -> Optional[int]:
    return _planning.auto_vllm_batch_size_for_pages(runtime_backend=runtime_backend, pages=pages)


def _flatten_lane_batches(lane: Dict[str, Any]) -> Dict[str, Any]:
    return _planning.flatten_lane_batches(lane)


def _utc_now_iso(now_ts: Optional[float] = None) -> str:
    return _runtime_support.utc_now_iso(now_ts)


def _parse_utc_iso(value: Optional[str]) -> Optional[float]:
    return _runtime_support.parse_utc_iso(value)


def _run_text_command(cmd: List[str]) -> str:
    return _runtime_support.run_text_command(cmd, subprocess_run_fn=subprocess.run)


def _process_group_members(pgid: int) -> List[int]:
    return _runtime_support.process_group_members(pgid, subprocess_run_fn=subprocess.run)


def _wait_for_process_group_exit(pgid: int, *, timeout_sec: float) -> bool:
    return _runtime_support.wait_for_process_group_exit(
        pgid,
        timeout_sec=timeout_sec,
        process_group_members_fn=_process_group_members,
    )


def _terminate_worker_process_group(worker: Dict[str, Any]) -> bool:
    return _runtime_support.terminate_worker_process_group(
        worker,
        killpg_fn=os.killpg,
        wait_for_process_group_exit_fn=lambda pgid: _wait_for_process_group_exit(pgid, timeout_sec=5.0),
        logger=LOGGER,
        signal_mod=signal,
    )


def _launch_worker_process(cmd: List[str], *, fh, env: Dict[str, str]) -> subprocess.Popen:
    return _runtime_support.launch_worker_process(
        cmd,
        fh=fh,
        env=env,
        subprocess_popen_fn=subprocess.Popen,
    )


def _parse_csv_table(text: str, columns: List[str]) -> List[Dict[str, str]]:
    return _runtime_support.parse_csv_table(text, columns)


def _collect_gpu_snapshot(*, visible_devices: List[int]) -> Dict[str, Any]:
    return _runtime_support.collect_gpu_snapshot(
        visible_devices=visible_devices,
        run_text_command_fn=_run_text_command,
    )


def _read_worker_runtime(runtime_path: Path) -> Dict[str, Any]:
    return _runtime_support.read_worker_runtime(runtime_path)


def _write_runtime_summary(*, runtime_dir: Path, db_path: Path) -> Path:
    return _runtime_support.write_runtime_summary(
        runtime_dir=runtime_dir,
        db_path=db_path,
        read_worker_runtime_fn=_read_worker_runtime,
        parse_utc_iso_fn=_parse_utc_iso,
        utc_now_iso_fn=_utc_now_iso,
        work_queue_counts_fn=work_queue_counts,
        iter_work_items_fn=iter_work_items,
    )


def _query_persistence_mode(*, visible_devices: List[int]) -> List[Dict[str, str]]:
    return _runtime_support.query_persistence_mode(
        visible_devices=visible_devices,
        run_text_command_fn=_run_text_command,
    )


def _ensure_gpu_preflight(*, visible_devices: List[int], mode: str) -> Dict[str, Any]:
    return _runtime_support.ensure_gpu_preflight(
        visible_devices=visible_devices,
        mode=mode,
        query_persistence_mode_fn=_query_persistence_mode,
        subprocess_run_fn=subprocess.run,
        utc_now_iso_fn=_utc_now_iso,
    )


def _collect_xid_faults(*, start_utc_iso: str) -> Dict[str, Any]:
    return _runtime_support.collect_xid_faults(
        start_utc_iso=start_utc_iso,
        run_text_command_fn=_run_text_command,
    )


def _start_gpu_telemetry(
    *,
    telemetry_path: Path,
    visible_devices: List[int],
    interval_sec: float,
    stop_event: threading.Event,
) -> threading.Thread:
    return _runtime_support.start_gpu_telemetry(
        telemetry_path=telemetry_path,
        visible_devices=visible_devices,
        interval_sec=interval_sec,
        stop_event=stop_event,
        collect_gpu_snapshot_fn=_collect_gpu_snapshot,
        logger=LOGGER,
    )


def _parse_shard_stem(stem: str) -> Optional[Dict[str, Any]]:
    return _reassembly.parse_shard_stem(stem)


def _split_markdown_pages(markdown_text: str, *, expected_pages: int) -> List[str]:
    return _reassembly.split_markdown_pages(
        markdown_text,
        expected_pages=expected_pages,
        split_page_outputs_fn=_split_page_outputs,
    )


def _archive_shard_artifact(*, out_root: Path, source_path: Path, relative_path: Path) -> None:
    return _reassembly.archive_shard_artifact(
        out_root=out_root,
        source_path=source_path,
        relative_path=relative_path,
    )


def _reassemble_canonical_output_for_source(
    *,
    out_root: Path,
    pdf_path: Path,
    source_name: str,
) -> bool:
    return _reassembly.reassemble_canonical_output_for_source(
        out_root=out_root,
        pdf_path=pdf_path,
        source_name=source_name,
        page_count_fn=_page_count,
        split_page_outputs_fn=_split_page_outputs,
        join_page_outputs_fn=_join_page_outputs,
        write_outputs_fn=_write_outputs,
    )


def _ensure_canonical_outputs(*, out_root: Path, pdf_root: Path, file_list: List[str]) -> None:
    return _reassembly.ensure_canonical_outputs(
        out_root=out_root,
        pdf_root=pdf_root,
        file_list=file_list,
        page_count_fn=_page_count,
        split_page_outputs_fn=_split_page_outputs,
        join_page_outputs_fn=_join_page_outputs,
        write_outputs_fn=_write_outputs,
    )



def _run_multi_cli(
    *,
    input_root: Path,
    out_root: Path,
    file_list: List[str],
    lane_devices: List[int],
    workers_per_gpu: int,
    model_root: Path,
    python_exe: Path,
    script_path: Path,
    max_pages: Optional[int],
    content_debug: bool,
    log_dir: Path,
    ocr_profile: str,
    prompt_override: Optional[str],
    attn_backend: str,
    base_size: Optional[int],
    image_size: Optional[int],
    crop_mode: Optional[bool],
    render_dpi: Optional[int],
    max_new_tokens: Optional[int],
    repetition_penalty: Optional[float],
    no_repeat_ngram_size: Optional[int],
    runtime_backend: str,
    vllm_batch_size: Optional[int],
    gpu_memory_utilization: Optional[float],
    disable_fp8_kv: bool,
    repair_mode: Optional[str],
    repair_exec_batch_target_pages: Optional[int],
    repair_exec_batch_target_items: Optional[int],
    scheduler: Optional[str],
    target_batch_pages: int,
    shard_pages: int,
    shard_threshold_pages: int,
) -> None:
    if str(runtime_backend or "").strip().lower() == "vllm":
        batches = _plan_work_batches(
            file_list=file_list,
            input_root=input_root,
            max_pages=max_pages,
            runtime_backend=runtime_backend,
            scheduler=scheduler,
            lane_devices=lane_devices,
            workers_per_gpu=workers_per_gpu,
            target_batch_pages=target_batch_pages,
            shard_pages=shard_pages,
            shard_threshold_pages=shard_threshold_pages,
        )
        if not batches:
            return

        log_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = out_root / "sidecars" / "ocr_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        work_db = runtime_dir / "work_queue.sqlite"
        init_work_db(work_db, batches=batches, replace=True)

        visible_devices = sorted({int(device) for device in lane_devices})
        preflight_mode = str(os.environ.get("GLOSSAPI_DEEPSEEK_GPU_PREFLIGHT", "ensure")).strip().lower()
        preflight = _ensure_gpu_preflight(visible_devices=visible_devices, mode=preflight_mode)
        (runtime_dir / "gpu_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")

        telemetry_path = runtime_dir / "gpu_telemetry.jsonl"
        with telemetry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "preflight", **preflight}) + "\n")
            fh.write(json.dumps({"kind": "initial_sample", **_collect_gpu_snapshot(visible_devices=visible_devices)}) + "\n")

        telemetry_stop = threading.Event()
        telemetry_thread = _start_gpu_telemetry(
            telemetry_path=telemetry_path,
            visible_devices=visible_devices,
            interval_sec=float(os.environ.get("GLOSSAPI_DEEPSEEK_TELEMETRY_INTERVAL_SEC", DEFAULT_TELEMETRY_INTERVAL_SEC)),
            stop_event=telemetry_stop,
        )
        stale_after_sec = float(os.environ.get("GLOSSAPI_DEEPSEEK_WORK_STALE_AFTER_SEC", DEFAULT_WORK_STALE_AFTER_SEC))
        heartbeat_sec = float(os.environ.get("GLOSSAPI_DEEPSEEK_WORK_HEARTBEAT_SEC", DEFAULT_WORK_HEARTBEAT_SEC))
        respawn_cap = int(os.environ.get("GLOSSAPI_DEEPSEEK_WORKER_RESPAWN_CAP", DEFAULT_WORKER_RESPAWN_CAP))
        work_max_attempts = int(
            max(1, int(os.environ.get("GLOSSAPI_DEEPSEEK_WORK_ITEM_MAX_ATTEMPTS", DEFAULT_WORK_ITEM_MAX_ATTEMPTS)))
        )
        xid_start = _utc_now_iso()

        def _start_worker(*, worker_id: str, visible_device: int, respawns: int) -> Dict[str, Any]:
            log_path = log_dir / f"{worker_id}.r{int(respawns)}.log"
            fh = log_path.open("w", encoding="utf-8")
            resolved_vllm_batch_size = (
                int(vllm_batch_size)
                if vllm_batch_size is not None
                else _auto_vllm_batch_size_for_pages(
                    runtime_backend=runtime_backend,
                    pages=int(target_batch_pages),
                )
            )
            cmd = _build_cli_command(
                input_dir=input_root,
                output_dir=out_root,
                files=[],
                page_ranges=None,
                model_dir=model_root,
                python_bin=python_exe,
                script=script_path,
                max_pages=max_pages,
                content_debug=content_debug,
                device="cuda",
                ocr_profile=ocr_profile,
                prompt_override=prompt_override,
                attn_backend=attn_backend,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                render_dpi=render_dpi,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                runtime_backend=runtime_backend,
                vllm_batch_size=resolved_vllm_batch_size,
                gpu_memory_utilization=gpu_memory_utilization,
                disable_fp8_kv=disable_fp8_kv,
                repair_mode=repair_mode,
                repair_exec_batch_target_pages=repair_exec_batch_target_pages,
                repair_exec_batch_target_items=repair_exec_batch_target_items,
                work_db=work_db,
                worker_id=worker_id,
                worker_runtime_file=runtime_dir / f"{worker_id}.runtime.json",
                work_stale_after_sec=stale_after_sec,
                work_heartbeat_sec=heartbeat_sec,
                work_max_attempts=work_max_attempts,
            )
            env = _build_env(python_bin=python_exe, visible_device=visible_device, script=script_path)
            LOGGER.info(
                "Running DeepSeek OCR worker=%s visible_gpu=%s batches=%d: %s",
                worker_id,
                visible_device,
                len(batches),
                " ".join(cmd),
            )
            proc = _launch_worker_process(cmd, fh=fh, env=env)
            return {
                "worker_id": worker_id,
                "visible_device": int(visible_device),
                "proc": proc,
                "fh": fh,
                "log_path": log_path,
                "respawns": int(respawns),
            }

        active_workers: List[Dict[str, Any]] = []
        worker_index = 0
        for visible_device in lane_devices:
            for _ in range(max(1, int(workers_per_gpu))):
                worker_id = f"worker_{worker_index:02d}_gpu{int(visible_device)}"
                active_workers.append(_start_worker(worker_id=worker_id, visible_device=int(visible_device), respawns=0))
                worker_index += 1

        failures: List[str] = []
        try:
            while active_workers:
                time.sleep(0.5)
                for worker in list(active_workers):
                    rc = worker["proc"].poll()
                    if rc is None:
                        continue
                    worker["fh"].close()
                    active_workers.remove(worker)
                    if int(rc) == 0:
                        continue
                    error_message = f"{worker['worker_id']} rc={int(rc)} log={worker['log_path']}"
                    LOGGER.warning("DeepSeek OCR worker failed: %s", error_message)
                    _terminate_worker_process_group(worker)
                    requeue_worker_batches(
                        work_db,
                        worker_id=str(worker["worker_id"]),
                        error=error_message,
                        max_attempts=work_max_attempts,
                    )
                    counts = work_queue_counts(work_db)
                    # Only respawn while there is retryable work left in the
                    # durable queue; terminally failed items should stop the run.
                    remaining_work = int(counts.get("pending", 0)) + int(counts.get("running", 0))
                    if remaining_work > 0 and int(worker["respawns"]) < respawn_cap:
                        active_workers.append(
                            _start_worker(
                                worker_id=str(worker["worker_id"]),
                                visible_device=int(worker["visible_device"]),
                                respawns=int(worker["respawns"]) + 1,
                            )
                        )
                        continue
                    failures.append(error_message)
            counts = work_queue_counts(work_db)
            if int(counts.get(STATUS_FAILED, 0)) > 0 or int(counts.get(STATUS_DONE, 0)) < int(counts.get("total", 0)):
                failures.append(f"incomplete_work queue_counts={counts}")
        finally:
            for worker in list(active_workers):
                _terminate_worker_process_group(worker)
                try:
                    worker["proc"].wait(timeout=5)
                except Exception:
                    pass
                worker["fh"].close()
            telemetry_stop.set()
            telemetry_thread.join(timeout=max(1.0, DEFAULT_TELEMETRY_INTERVAL_SEC))
            with telemetry_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": "final_sample", **_collect_gpu_snapshot(visible_devices=visible_devices)}) + "\n")
                fh.write(json.dumps({"kind": "xid_faults", **_collect_xid_faults(start_utc_iso=xid_start)}) + "\n")
            _write_runtime_summary(runtime_dir=runtime_dir, db_path=work_db)

        if failures:
            raise RuntimeError("DeepSeek OCR multi-worker failure(s): " + "; ".join(failures))
        return

    lanes = _plan_lane_batches(
        file_list=file_list,
        input_root=input_root,
        lane_devices=lane_devices,
        workers_per_gpu=workers_per_gpu,
        max_pages=max_pages,
        runtime_backend=runtime_backend,
        scheduler=scheduler,
        target_batch_pages=target_batch_pages,
        shard_pages=shard_pages,
        shard_threshold_pages=shard_threshold_pages,
    )
    if not lanes:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    with ExitStack() as stack:
        procs = []

        for lane in lanes:
            lane_id = int(lane["lane_id"])
            visible_device = int(lane["visible_device"])
            lane_plan = _flatten_lane_batches(lane)
            files = list(lane_plan["files"])
            page_ranges = list(lane_plan["page_ranges"])
            pages = int(lane_plan["pages"])
            if pages <= 0:
                continue
            resolved_vllm_batch_size = (
                int(vllm_batch_size)
                if vllm_batch_size is not None
                else _auto_vllm_batch_size_for_pages(
                    runtime_backend=runtime_backend,
                    pages=min(int(target_batch_pages), int(pages)),
                )
            )
            log_path = log_dir / f"lane_{lane_id:02d}_gpu{visible_device}.log"
            fh = stack.enter_context(log_path.open("w", encoding="utf-8"))
            cmd = _build_cli_command(
                input_dir=input_root,
                output_dir=out_root,
                files=files,
                page_ranges=page_ranges,
                model_dir=model_root,
                python_bin=python_exe,
                script=script_path,
                max_pages=max_pages,
                content_debug=content_debug,
                device="cuda",
                ocr_profile=ocr_profile,
                prompt_override=prompt_override,
                attn_backend=attn_backend,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                render_dpi=render_dpi,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                runtime_backend=runtime_backend,
                vllm_batch_size=resolved_vllm_batch_size,
                gpu_memory_utilization=gpu_memory_utilization,
                disable_fp8_kv=disable_fp8_kv,
                repair_mode=repair_mode,
                repair_exec_batch_target_pages=repair_exec_batch_target_pages,
                repair_exec_batch_target_items=repair_exec_batch_target_items,
            )
            env = _build_env(python_bin=python_exe, visible_device=visible_device, script=script_path)
            LOGGER.info(
                "Running DeepSeek OCR lane=%s visible_gpu=%s pages=%s planned_batches=%s files=%d ranges=%d: %s",
                lane_id,
                visible_device,
                pages,
                lane_plan["planned_batch_count"],
                len(files),
                len(page_ranges),
                " ".join(cmd),
            )
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)  # nosec: controlled args
            procs.append((lane, log_path, proc))

        for lane, log_path, proc in procs:
            rc = proc.wait()
            if rc != 0:
                failures.append(
                    f"lane={lane['lane_id']} gpu={lane['visible_device']} rc={rc} log={log_path}"
                )
    if failures:
        raise RuntimeError("DeepSeek OCR multi-worker failure(s): " + "; ".join(failures))


def run_for_files(
    self_ref: Any,
    files: Iterable[str],
    *,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,  # kept for API compatibility
    max_pages: Optional[int] = None,
    allow_stub: bool = False,  # ignored after stub removal; kept for compatibility
    allow_cli: bool = True,  # ignored after stub removal; kept for compatibility
    python_bin: Optional[Path] = None,
    vllm_script: Optional[Path] = None,
    content_debug: bool = False,
    persist_engine: bool = True,  # placeholder for future session reuse
    precision: Optional[str] = None,  # reserved
    device: Optional[str] = None,
    runtime_backend: str = DEFAULT_RUNTIME_BACKEND,
    ocr_profile: str = DEFAULT_OCR_PROFILE,
    prompt_override: Optional[str] = None,
    attn_backend: str = DEFAULT_ATTN_BACKEND,
    base_size: Optional[int] = None,
    image_size: Optional[int] = None,
    crop_mode: Optional[bool] = None,
    render_dpi: Optional[int] = DEFAULT_RENDER_DPI,
    max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS,
    repetition_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    use_gpus: Optional[str] = None,
    devices: Optional[List[int]] = None,
    workers_per_gpu: int = DEFAULT_WORKERS_PER_GPU,
    gpu_memory_utilization: Optional[float] = DEFAULT_GPU_MEMORY_UTILIZATION,
    disable_fp8_kv: bool = False,
    vllm_batch_size: Optional[int] = None,
    repair_mode: str = DEFAULT_REPAIR_MODE,
    repair_exec_batch_target_pages: Optional[int] = None,
    repair_exec_batch_target_items: Optional[int] = None,
    scheduler: str = "auto",
    target_batch_pages: int = AUTO_VLLM_BATCH_PAGE_CAP,
    shard_pages: int = 0,
    shard_threshold_pages: int = 0,
    **_: Any,
) -> Dict[str, Any]:
    """Run DeepSeek OCR for the provided files."""

    requested_stub = bool(allow_stub)
    del allow_stub, allow_cli, persist_engine, precision
    if requested_stub or os.environ.get("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "0") == "1":
        raise RuntimeError(
            "DeepSeek stub execution has been removed. "
            "Unset GLOSSAPI_DEEPSEEK_ALLOW_STUB and configure the real DeepSeek runtime."
        )

    runtime_backend_norm = str(
        runtime_backend or os.environ.get("GLOSSAPI_DEEPSEEK_RUNTIME_BACKEND", DEFAULT_RUNTIME_BACKEND)
    ).strip().lower()
    if runtime_backend_norm not in {"transformers", "vllm"}:
        raise ValueError("runtime_backend must be 'transformers' or 'vllm'")
    resolved_render_dpi = resolve_render_dpi(render_dpi)
    resolved_max_new_tokens = int(DEFAULT_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens)
    resolved_gpu_memory_utilization = resolve_gpu_memory_utilization(gpu_memory_utilization)
    resolved_repair_mode = str(repair_mode or DEFAULT_REPAIR_MODE)

    file_list = [str(f) for f in files or []]
    if not file_list:
        return {}

    input_root = Path(getattr(self_ref, "input_dir", ".")).resolve()
    pdf_root = (input_root / "downloads") if (input_root / "downloads").exists() else input_root
    out_root = Path(output_dir) if output_dir else Path(getattr(self_ref, "output_dir", input_root))
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_root = Path(
        model_dir
        or os.environ.get("GLOSSAPI_DEEPSEEK_MODEL_DIR", "")
        or (REPO_ROOT / "deepseek-ocr-2-model" / "DeepSeek-OCR-2")
    )
    if not model_root.exists():
        raise FileNotFoundError(
            "DeepSeek model directory not found. Set model_dir or GLOSSAPI_DEEPSEEK_MODEL_DIR."
        )

    default_script = DEFAULT_VLLM_SCRIPT if runtime_backend_norm == "vllm" else DEFAULT_SCRIPT
    script_path = Path(
        vllm_script
        or os.environ.get("GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT", "")
        or default_script
    )
    if not script_path.exists():
        raise FileNotFoundError(f"DeepSeek OCR runner script not found: {script_path}")

    python_exe = resolve_deepseek_python(explicit_python=python_bin)
    if not python_exe.exists():
        raise FileNotFoundError(f"DeepSeek Python interpreter not found: {python_exe}")

    lane_devices = _resolve_lane_devices(
        use_gpus=use_gpus,
        devices=devices,
        workers_per_gpu=int(max(1, workers_per_gpu)),
        device=device,
    )
    multi_requested = str(use_gpus or "single").strip().lower() == "multi" or int(max(1, workers_per_gpu)) > 1
    if multi_requested and lane_devices:
        _run_multi_cli(
            input_root=pdf_root,
            out_root=out_root,
            file_list=file_list,
            lane_devices=lane_devices,
            workers_per_gpu=int(max(1, workers_per_gpu)),
            model_root=model_root,
            python_exe=python_exe,
            script_path=script_path,
            max_pages=max_pages,
            content_debug=content_debug,
            log_dir=Path(log_dir) if log_dir else (out_root / "logs" / "deepseek_workers"),
            ocr_profile=ocr_profile,
            prompt_override=prompt_override,
            attn_backend=attn_backend,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            render_dpi=resolved_render_dpi,
            max_new_tokens=resolved_max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            runtime_backend=runtime_backend_norm,
            vllm_batch_size=vllm_batch_size,
            gpu_memory_utilization=resolved_gpu_memory_utilization,
            disable_fp8_kv=disable_fp8_kv,
            repair_mode=resolved_repair_mode,
            repair_exec_batch_target_pages=repair_exec_batch_target_pages,
            repair_exec_batch_target_items=repair_exec_batch_target_items,
            scheduler=scheduler,
            target_batch_pages=int(max(1, target_batch_pages)),
            shard_pages=int(max(0, shard_pages)),
            shard_threshold_pages=int(max(0, shard_threshold_pages)),
        )
    else:
        resolved_vllm_batch_size = (
            int(vllm_batch_size)
            if vllm_batch_size is not None
            else _auto_vllm_batch_size(
                runtime_backend=runtime_backend_norm,
                file_list=file_list,
                input_root=pdf_root,
                max_pages=max_pages,
            )
        )
        _run_cli(
            input_dir=pdf_root,
            output_dir=out_root,
            files=file_list,
            page_ranges=None,
            model_dir=model_root,
            python_bin=python_exe,
            script=script_path,
            max_pages=max_pages,
            content_debug=content_debug,
            device=device,
            ocr_profile=ocr_profile,
            prompt_override=prompt_override,
            attn_backend=attn_backend,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            render_dpi=resolved_render_dpi,
            max_new_tokens=resolved_max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            runtime_backend=runtime_backend_norm,
            vllm_batch_size=resolved_vllm_batch_size,
            gpu_memory_utilization=resolved_gpu_memory_utilization,
            disable_fp8_kv=disable_fp8_kv,
            repair_mode=resolved_repair_mode,
            repair_exec_batch_target_pages=repair_exec_batch_target_pages,
            repair_exec_batch_target_items=repair_exec_batch_target_items,
        )

    _ensure_canonical_outputs(out_root=out_root, pdf_root=pdf_root, file_list=file_list)

    results: Dict[str, Any] = {}
    for name in file_list:
        pdf_path = (pdf_root / name).resolve()
        stem = Path(name).stem
        md_path = md_dir / f"{stem}.md"
        metrics_path = metrics_dir / f"{stem}.metrics.json"
        if not md_path.exists():
            raise FileNotFoundError(f"DeepSeek OCR did not produce markdown for {name}: {md_path}")
        if not md_path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"DeepSeek OCR produced empty markdown for {name}: {md_path}")
        page_count = _page_count(pdf_path)
        if metrics_path.exists():
            try:
                results[stem] = json.loads(metrics_path.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        results[stem] = {"page_count": page_count}
        metrics_path.write_text(json.dumps(results[stem], indent=2), encoding="utf-8")

    return results
