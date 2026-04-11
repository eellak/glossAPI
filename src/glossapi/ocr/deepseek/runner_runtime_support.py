"""Runtime-support helpers for the DeepSeek runner shim."""

from __future__ import annotations

import calendar
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def utc_now_iso(now_ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(now_ts) if now_ts is not None else time.time()))


def parse_utc_iso(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(calendar.timegm(time.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ")))
    except Exception:
        return None


def run_text_command(
    cmd: List[str],
    *,
    subprocess_run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    proc = subprocess_run_fn(cmd, check=True, capture_output=True, text=True)  # nosec: controlled args
    return str(proc.stdout or "").strip()


def parse_csv_table(text: str, columns: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < len(columns):
            parts.extend([""] * (len(columns) - len(parts)))
        rows.append({name: str(parts[idx]) for idx, name in enumerate(columns)})
    return rows


def collect_gpu_snapshot(
    *,
    visible_devices: List[int],
    run_text_command_fn: Callable[[List[str]], str],
) -> Dict[str, Any]:
    gpu_text = run_text_command_fn(
        [
            "nvidia-smi",
            f"--id={','.join(str(device) for device in visible_devices)}",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,persistence_mode",
            "--format=csv,noheader,nounits",
        ]
    )
    process_text = run_text_command_fn(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "captured_at": utc_now_iso(),
        "gpus": parse_csv_table(
            gpu_text,
            [
                "index",
                "name",
                "utilization_gpu",
                "memory_used_mib",
                "memory_total_mib",
                "temperature_c",
                "power_draw_w",
                "persistence_mode",
            ],
        ),
        "processes": parse_csv_table(
            process_text,
            [
                "gpu_uuid",
                "pid",
                "process_name",
                "used_memory_mib",
            ],
        ),
    }


def read_worker_runtime(runtime_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(runtime_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_runtime_summary(
    *,
    runtime_dir: Path,
    db_path: Path,
    read_worker_runtime_fn: Callable[[Path], Dict[str, Any]],
    parse_utc_iso_fn: Callable[[Optional[str]], Optional[float]],
    utc_now_iso_fn: Callable[[Optional[float]], str],
    work_queue_counts_fn: Callable[[Path], Dict[str, Any]],
    iter_work_items_fn: Callable[[Path], Any],
) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    first_batch_started = []
    last_batch_finished = []
    engine_ready = []
    for path in sorted(runtime_dir.glob("worker_*.runtime.json")):
        data = read_worker_runtime_fn(path)
        workers.append(data)
        first_batch_started_ts = parse_utc_iso_fn(data.get("first_batch_started_at"))
        last_batch_finished_ts = parse_utc_iso_fn(data.get("last_batch_finished_at"))
        engine_ready_ts = parse_utc_iso_fn(data.get("engine_ready_at"))
        if first_batch_started_ts is not None:
            first_batch_started.append(first_batch_started_ts)
        if last_batch_finished_ts is not None:
            last_batch_finished.append(last_batch_finished_ts)
        if engine_ready_ts is not None:
            engine_ready.append(engine_ready_ts)
    steady_summary = {
        "first_batch_started_at": utc_now_iso_fn(min(first_batch_started)) if first_batch_started else None,
        "last_batch_finished_at": utc_now_iso_fn(max(last_batch_finished)) if last_batch_finished else None,
        "all_workers_ready_at": utc_now_iso_fn(max(engine_ready)) if engine_ready else None,
        "first_batch_to_last_batch_window_sec": (
            float(max(last_batch_finished) - min(first_batch_started))
            if first_batch_started and last_batch_finished
            else None
        ),
        "all_workers_ready_to_last_batch_window_sec": (
            float(max(last_batch_finished) - max(engine_ready))
            if engine_ready and last_batch_finished
            else None
        ),
    }
    summary_path = runtime_dir / "runtime_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now_iso_fn(),
                "queue_counts": work_queue_counts_fn(db_path),
                "work_items": list(iter_work_items_fn(db_path)),
                "workers": workers,
                "steady_state": steady_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def query_persistence_mode(
    *,
    visible_devices: List[int],
    run_text_command_fn: Callable[[List[str]], str],
) -> List[Dict[str, str]]:
    raw = run_text_command_fn(
        [
            "nvidia-smi",
            f"--id={','.join(str(device) for device in visible_devices)}",
            "--query-gpu=index,persistence_mode",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_csv_table(raw, ["index", "persistence_mode"])


def ensure_gpu_preflight(
    *,
    visible_devices: List[int],
    mode: str,
    query_persistence_mode_fn: Callable[..., List[Dict[str, str]]],
    subprocess_run_fn: Callable[..., Any],
    utc_now_iso_fn: Callable[[Optional[float]], str],
) -> Dict[str, Any]:
    mode_norm = str(mode or "warn").strip().lower()
    status = {
        "mode": mode_norm,
        "checked_at": utc_now_iso_fn(),
        "before": query_persistence_mode_fn(visible_devices=visible_devices),
        "changed": False,
    }
    disabled = [item for item in status["before"] if str(item.get("persistence_mode", "")).lower() != "enabled"]
    if not disabled or mode_norm == "off":
        status["after"] = list(status["before"])
        return status
    if mode_norm == "ensure":
        try:
            subprocess_run_fn(["sudo", "-n", "nvidia-smi", "-pm", "1"], check=True, capture_output=True, text=True)  # nosec: controlled args
            status["changed"] = True
        except Exception as exc:
            status["ensure_error"] = str(exc)
    status["after"] = query_persistence_mode_fn(visible_devices=visible_devices)
    return status


def collect_xid_faults(
    *,
    start_utc_iso: str,
    run_text_command_fn: Callable[[List[str]], str],
) -> Dict[str, Any]:
    cmd = [
        "journalctl",
        "-k",
        "--since",
        str(start_utc_iso),
        "--no-pager",
    ]
    try:
        output = run_text_command_fn(cmd)
    except Exception as exc:
        return {
            "supported": False,
            "error": str(exc),
            "faults": [],
        }
    faults = [line for line in output.splitlines() if "NVRM: Xid" in line]
    return {
        "supported": True,
        "faults": faults,
    }


def start_gpu_telemetry(
    *,
    telemetry_path: Path,
    visible_devices: List[int],
    interval_sec: float,
    stop_event: threading.Event,
    collect_gpu_snapshot_fn: Callable[..., Dict[str, Any]],
    logger,
) -> threading.Thread:
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    def _loop() -> None:
        while not stop_event.wait(float(max(1.0, interval_sec))):
            try:
                with telemetry_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"kind": "sample", **collect_gpu_snapshot_fn(visible_devices=visible_devices)}) + "\n")
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.warning("GPU telemetry sample failed: %s", exc)

    thread = threading.Thread(target=_loop, name="deepseek-gpu-telemetry", daemon=True)
    thread.start()
    return thread


def process_group_members(
    pgid: int,
    *,
    subprocess_run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> List[int]:
    proc = subprocess_run_fn(["pgrep", "-g", str(int(pgid))], check=False, capture_output=True, text=True)  # nosec: controlled args
    if int(proc.returncode) not in {0, 1}:
        return []
    members: List[int] = []
    for line in str(proc.stdout or "").splitlines():
        line = line.strip()
        if line:
            try:
                members.append(int(line))
            except ValueError:
                continue
    return members


def wait_for_process_group_exit(
    pgid: int,
    *,
    timeout_sec: float,
    process_group_members_fn: Callable[[int], List[int]],
) -> bool:
    deadline = time.time() + float(max(0.0, timeout_sec))
    while time.time() <= deadline:
        if not process_group_members_fn(pgid):
            return True
        time.sleep(0.2)
    return not process_group_members_fn(pgid)


def terminate_worker_process_group(
    worker: Dict[str, Any],
    *,
    killpg_fn: Callable[[int, int], None],
    wait_for_process_group_exit_fn: Callable[[int], bool],
    logger,
    signal_mod=signal,
) -> bool:
    pgid = int(worker["proc"].pid)
    worker_id = str(worker["worker_id"])
    for sig, _grace_sec in ((signal_mod.SIGTERM, 5.0), (signal_mod.SIGKILL, 5.0)):
        try:
            killpg_fn(pgid, sig)
        except ProcessLookupError:
            return True
        except Exception as exc:
            logger.warning("Failed to signal worker process group %s pgid=%s: %s", worker_id, pgid, exc)
            return False
        if wait_for_process_group_exit_fn(pgid):
            return True
    logger.warning("Worker process group %s pgid=%s did not exit cleanly", worker_id, pgid)
    return False


def launch_worker_process(
    cmd: List[str],
    *,
    fh,
    env: Dict[str, str],
    subprocess_popen_fn: Callable[..., Any] = subprocess.Popen,
):
    return subprocess_popen_fn(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )  # nosec: controlled args
