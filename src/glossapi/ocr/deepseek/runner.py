"""DeepSeek OCR runner."""

from __future__ import annotations

from contextlib import ExitStack
import calendar
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from glossapi.ocr.deepseek.scheduling import (
    SourceDocument,
    assign_batches_to_lanes,
    build_exact_fill_batches,
    build_fixed_shard_slices,
    build_whole_document_slices,
    pack_slices_into_batches,
)
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

try:
    import pypdfium2 as _pypdfium2
except Exception:  # pragma: no cover - optional dependency
    _pypdfium2 = None

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCRIPT = REPO_ROOT / "src" / "glossapi" / "ocr" / "deepseek" / "run_pdf_ocr_transformers.py"
DEFAULT_VLLM_SCRIPT = REPO_ROOT / "src" / "glossapi" / "ocr" / "deepseek" / "run_pdf_ocr_vllm.py"
AUTO_VLLM_BATCH_PAGE_CAP = 160
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_WORKER_RESPAWN_CAP = 3
DEFAULT_WORK_ITEM_MAX_ATTEMPTS = 2
DEFAULT_WORK_STALE_AFTER_SEC = 900.0
DEFAULT_WORK_HEARTBEAT_SEC = 10.0
DEFAULT_TELEMETRY_INTERVAL_SEC = 15.0
SHARD_STEM_RE = re.compile(r"^(?P<source_stem>.+)__p(?P<start>\d{5})-(?P<end>\d{5})$")
REASSEMBLED_CONFIG_KEYS = (
    "ocr_profile",
    "attn_backend",
    "runtime_backend",
    "base_size",
    "image_size",
    "crop_mode",
    "render_dpi",
    "max_new_tokens",
    "batch_size",
    "gpu_memory_utilization",
    "disable_fp8_kv",
    "repair_mode",
)


def _page_count(pdf_path: Path) -> int:
    if _pypdfium2 is None:
        return 0
    try:
        return len(_pypdfium2.PdfDocument(str(pdf_path)))
    except Exception:
        return 0


def _build_cli_command(
    input_dir: Path,
    output_dir: Path,
    *,
    files: List[str],
    page_ranges: Optional[List[str]],
    model_dir: Path,
    python_bin: Optional[Path],
    script: Path,
    max_pages: Optional[int],
    content_debug: bool,
    device: Optional[str],
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
    repair_exec_batch_target_pages: Optional[int] = None,
    repair_exec_batch_target_items: Optional[int] = None,
    work_db: Optional[Path] = None,
    worker_id: Optional[str] = None,
    worker_runtime_file: Optional[Path] = None,
    work_stale_after_sec: Optional[float] = None,
    work_heartbeat_sec: Optional[float] = None,
    work_max_attempts: Optional[int] = None,
) -> List[str]:
    python_exe = Path(python_bin) if python_bin else Path(sys.executable)
    cmd: List[str] = [
        str(python_exe),
        str(script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--model-dir",
        str(model_dir),
    ]
    if files:
        cmd += ["--files", *files]
    if page_ranges:
        cmd += ["--page-ranges", *page_ranges]
    if max_pages is not None:
        cmd += ["--max-pages", str(max_pages)]
    if content_debug:
        cmd.append("--content-debug")
    if device:
        cmd += ["--device", str(device)]
    if ocr_profile:
        cmd += ["--ocr-profile", str(ocr_profile)]
    if prompt_override:
        cmd += ["--prompt-override", str(prompt_override)]
    if attn_backend:
        cmd += ["--attn-backend", str(attn_backend)]
    if base_size is not None:
        cmd += ["--base-size", str(int(base_size))]
    if image_size is not None:
        cmd += ["--image-size", str(int(image_size))]
    if crop_mode is True:
        cmd.append("--crop-mode")
    elif crop_mode is False:
        cmd.append("--no-crop-mode")
    if render_dpi is not None:
        cmd += ["--render-dpi", str(int(render_dpi))]
    if max_new_tokens is not None:
        cmd += ["--max-new-tokens", str(int(max_new_tokens))]
    if work_db is not None:
        cmd += ["--work-db", str(work_db)]
    if worker_id:
        cmd += ["--worker-id", str(worker_id)]
    if worker_runtime_file is not None:
        cmd += ["--worker-runtime-file", str(worker_runtime_file)]
    if work_stale_after_sec is not None:
        cmd += ["--work-stale-after-sec", str(float(work_stale_after_sec))]
    if work_heartbeat_sec is not None:
        cmd += ["--work-heartbeat-sec", str(float(work_heartbeat_sec))]
    if work_max_attempts is not None:
        cmd += ["--work-max-attempts", str(int(work_max_attempts))]
    if repetition_penalty is not None:
        cmd += ["--repetition-penalty", str(float(repetition_penalty))]
    if no_repeat_ngram_size is not None:
        cmd += ["--no-repeat-ngram-size", str(int(no_repeat_ngram_size))]
    runtime_backend_norm = str(runtime_backend or "transformers").strip().lower()
    if runtime_backend_norm == "vllm":
        if vllm_batch_size is not None:
            cmd += ["--batch-size", str(int(vllm_batch_size))]
        if gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(float(gpu_memory_utilization))]
        if disable_fp8_kv:
            cmd.append("--disable-fp8-kv")
        if repair_mode:
            cmd += ["--repair-mode", str(repair_mode)]
        if repair_exec_batch_target_pages is not None:
            cmd += ["--repair-exec-batch-target-pages", str(int(repair_exec_batch_target_pages))]
        if repair_exec_batch_target_items is not None:
            cmd += ["--repair-exec-batch-target-items", str(int(repair_exec_batch_target_items))]
    return cmd


def _build_env(
    *,
    python_bin: Optional[Path],
    visible_device: Optional[int] = None,
    script: Optional[Path] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    if python_bin:
        python_path = Path(python_bin).expanduser()
        venv_bin = str(python_path.parent)
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(python_path.parent.parent)
    if script is not None:
        script_path = Path(script).expanduser().resolve()
        src_root = next((parent for parent in script_path.parents if (parent / "glossapi").is_dir()), None)
        if src_root is not None:
            src_root_str = str(src_root)
            existing_pythonpath = str(env.get("PYTHONPATH", "")).strip()
            pythonpath_entries = [src_root_str]
            if existing_pythonpath:
                pythonpath_entries.extend(
                    entry
                    for entry in existing_pythonpath.split(os.pathsep)
                    if entry and entry != src_root_str
                )
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.pop("PYTHONHOME", None)
    if visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    if shutil.which("cc1plus", path=env.get("PATH", "")) is None:
        for candidate in sorted(Path("/usr/lib/gcc/x86_64-linux-gnu").glob("*/cc1plus")):
            env["PATH"] = f"{candidate.parent}:{env.get('PATH', '')}"
            break
    ld_entries: List[str] = []
    if python_bin:
        # Keep the venv path semantics instead of resolving the interpreter symlink
        # back to `/usr/bin/python...`; the wheel-managed CUDA libs live under the
        # virtualenv tree, not under the system interpreter location.
        venv_root = Path(python_bin).expanduser().parent.parent
        for site_packages in sorted((venv_root / "lib").glob("python*/site-packages")):
            nvidia_root = site_packages / "nvidia"
            if not nvidia_root.is_dir():
                continue
            for lib_dir in sorted(nvidia_root.glob("*/lib")):
                if lib_dir.is_dir():
                    ld_entries.append(str(lib_dir))
    ld_path = env.get("GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH")
    if ld_path:
        ld_entries.extend(entry for entry in str(ld_path).split(os.pathsep) if entry)
    existing_ld = str(env.get("LD_LIBRARY_PATH", "")).strip()
    if existing_ld:
        ld_entries.extend(entry for entry in existing_ld.split(os.pathsep) if entry)
    if ld_entries:
        deduped: List[str] = []
        seen: Set[str] = set()
        for entry in ld_entries:
            if entry and entry not in seen:
                seen.add(entry)
                deduped.append(entry)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(deduped)
    return env


def _run_cli(
    input_dir: Path,
    output_dir: Path,
    *,
    files: List[str],
    model_dir: Path,
    python_bin: Optional[Path],
    script: Path,
    max_pages: Optional[int],
    content_debug: bool,
    device: Optional[str],
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
    visible_device: Optional[int] = None,
) -> None:
    cmd = _build_cli_command(
        input_dir=input_dir,
        output_dir=output_dir,
        files=files,
        page_ranges=None,
        model_dir=model_dir,
        python_bin=python_bin,
        script=script,
        max_pages=max_pages,
        content_debug=content_debug,
        device=device,
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
        vllm_batch_size=vllm_batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_fp8_kv=disable_fp8_kv,
        repair_mode=repair_mode,
        repair_exec_batch_target_pages=repair_exec_batch_target_pages,
        repair_exec_batch_target_items=repair_exec_batch_target_items,
    )
    env = _build_env(python_bin=python_bin, visible_device=visible_device, script=script)

    LOGGER.info("Running DeepSeek OCR CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # nosec: controlled arguments


def _parse_device_index(device: Optional[str]) -> Optional[int]:
    if not device:
        return None
    value = str(device).strip().lower()
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _detect_visible_gpus() -> List[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        parsed = [piece.strip() for piece in visible.split(",") if piece.strip()]
        if parsed and all(piece.isdigit() for piece in parsed):
            return [int(piece) for piece in parsed]
    torch_mod = None
    try:  # pragma: no cover - best effort
        import torch as torch_mod  # type: ignore
    except Exception:  # pragma: no cover - optional import
        torch_mod = None
    if torch_mod is not None:
        try:
            if torch_mod.cuda.is_available():
                return list(range(torch_mod.cuda.device_count()))
        except Exception:
            pass
    try:  # pragma: no cover - shell fallback
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        devices: List[int] = []
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("GPU "):
                    prefix = line.split(":", 1)[0]
                    idx = prefix.split()[1]
                    if idx.isdigit():
                        devices.append(int(idx))
        return devices
    except Exception:
        return []


def _resolve_lane_devices(
    *,
    use_gpus: Optional[str],
    devices: Optional[List[int]],
    workers_per_gpu: int,
    device: Optional[str],
) -> List[int]:
    if devices:
        resolved = [int(dev) for dev in devices]
        if resolved:
            return resolved
    if str(use_gpus or "single").strip().lower() == "multi":
        resolved = _detect_visible_gpus()
        if resolved:
            return resolved
    if workers_per_gpu > 1:
        from_device = _parse_device_index(device)
        if from_device is not None:
            return [from_device]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            first = visible.split(",", 1)[0].strip()
            if first.isdigit():
                return [int(first)]
        return [0]
    return []


def _effective_page_count(pdf_path: Path, max_pages: Optional[int]) -> int:
    count = _page_count(pdf_path)
    if max_pages is not None and count > 0:
        return min(count, int(max_pages))
    return max(1, count)


def _source_documents(
    *,
    file_list: List[str],
    input_root: Path,
    max_pages: Optional[int],
) -> List[SourceDocument]:
    documents: List[SourceDocument] = []
    for name in file_list:
        pdf_path = (input_root / name).resolve()
        documents.append(
            SourceDocument(
                name=str(name),
                pages=int(_effective_page_count(pdf_path, max_pages)),
            )
        )
    return documents


def _plan_lanes(
    *,
    file_list: List[str],
    input_root: Path,
    lane_devices: List[int],
    workers_per_gpu: int,
    max_pages: Optional[int],
) -> List[Dict[str, Any]]:
    lanes: List[Dict[str, Any]] = []
    lane_id = 0
    for visible_device in lane_devices:
        for _ in range(max(1, int(workers_per_gpu))):
            lanes.append(
                {
                    "lane_id": lane_id,
                    "visible_device": int(visible_device),
                    "files": [],
                    "weight": 0,
                }
            )
            lane_id += 1
    if not lanes:
        return []

    weighted_files = []
    for name in file_list:
        pdf_path = (input_root / name).resolve()
        weighted_files.append((name, _effective_page_count(pdf_path, max_pages)))
    weighted_files.sort(key=lambda item: (-item[1], item[0]))

    for name, weight in weighted_files:
        lane = min(lanes, key=lambda item: int(item["weight"]))
        lane["files"].append(name)
        lane["weight"] = int(lane["weight"]) + int(weight)
    return lanes


def _resolve_scheduler(
    *,
    scheduler: Optional[str],
    runtime_backend: str,
    lane_devices: List[int],
    workers_per_gpu: int,
) -> str:
    scheduler_norm = str(scheduler or "auto").strip().lower()
    if scheduler_norm not in {"auto", "whole_doc", "fixed_shard", "exact_fill"}:
        raise ValueError("scheduler must be one of 'auto', 'whole_doc', 'fixed_shard', or 'exact_fill'")
    if scheduler_norm != "auto":
        return scheduler_norm
    runtime_backend_norm = str(runtime_backend or "transformers").strip().lower()
    lane_count = max(1, len(lane_devices)) * max(1, int(workers_per_gpu))
    if runtime_backend_norm == "vllm" and lane_count > 1:
        return "exact_fill"
    return "whole_doc"


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
    documents = _source_documents(
        file_list=file_list,
        input_root=input_root,
        max_pages=max_pages,
    )
    scheduler_norm = _resolve_scheduler(
        scheduler=scheduler,
        runtime_backend=runtime_backend,
        lane_devices=lane_devices,
        workers_per_gpu=workers_per_gpu,
    )
    if scheduler_norm == "exact_fill":
        batches = build_exact_fill_batches(
            documents,
            target_batch_pages=max(1, int(target_batch_pages)),
        )
    else:
        if scheduler_norm == "fixed_shard":
            slices = build_fixed_shard_slices(
                documents,
                shard_pages=max(1, int(shard_pages)),
                shard_threshold_pages=max(0, int(shard_threshold_pages)),
            )
        else:
            slices = build_whole_document_slices(documents)
        batches = pack_slices_into_batches(
            slices,
            target_batch_pages=max(1, int(target_batch_pages)),
        )
    lanes = assign_batches_to_lanes(
        batches,
        devices=lane_devices,
        workers_per_gpu=workers_per_gpu,
    )
    return [lane.to_dict() for lane in lanes if lane.batches]


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
    documents = _source_documents(
        file_list=file_list,
        input_root=input_root,
        max_pages=max_pages,
    )
    scheduler_norm = _resolve_scheduler(
        scheduler=scheduler,
        runtime_backend=runtime_backend,
        lane_devices=lane_devices,
        workers_per_gpu=workers_per_gpu,
    )
    if scheduler_norm == "exact_fill":
        batches = build_exact_fill_batches(
            documents,
            target_batch_pages=max(1, int(target_batch_pages)),
        )
    else:
        if scheduler_norm == "fixed_shard":
            slices = build_fixed_shard_slices(
                documents,
                shard_pages=max(1, int(shard_pages)),
                shard_threshold_pages=max(0, int(shard_threshold_pages)),
            )
        else:
            slices = build_whole_document_slices(documents)
        batches = pack_slices_into_batches(
            slices,
            target_batch_pages=max(1, int(target_batch_pages)),
        )
    return [batch.to_dict() for batch in batches if int(batch.pages) > 0]


def _auto_vllm_batch_size(
    *,
    runtime_backend: str,
    file_list: List[str],
    input_root: Path,
    max_pages: Optional[int],
) -> Optional[int]:
    if str(runtime_backend or "").strip().lower() != "vllm":
        return None
    total_pages = 0
    for name in file_list:
        pdf_path = (input_root / name).resolve()
        total_pages += int(_effective_page_count(pdf_path, max_pages))
    if total_pages <= 0:
        return 1
    return min(int(total_pages), int(AUTO_VLLM_BATCH_PAGE_CAP))


def _auto_vllm_batch_size_for_pages(*, runtime_backend: str, pages: int) -> Optional[int]:
    if str(runtime_backend or "").strip().lower() != "vllm":
        return None
    if int(pages) <= 0:
        return 1
    return min(int(pages), int(AUTO_VLLM_BATCH_PAGE_CAP))


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


def _utc_now_iso(now_ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(now_ts) if now_ts is not None else time.time()))


def _parse_utc_iso(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(calendar.timegm(time.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ")))
    except Exception:
        return None


def _run_text_command(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)  # nosec: controlled args
    return str(proc.stdout or "").strip()


def _process_group_members(pgid: int) -> List[int]:
    proc = subprocess.run(["pgrep", "-g", str(int(pgid))], check=False, capture_output=True, text=True)  # nosec: controlled args
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


def _wait_for_process_group_exit(pgid: int, *, timeout_sec: float) -> bool:
    deadline = time.time() + float(max(0.0, timeout_sec))
    while time.time() <= deadline:
        if not _process_group_members(pgid):
            return True
        time.sleep(0.2)
    return not _process_group_members(pgid)


def _terminate_worker_process_group(worker: Dict[str, Any]) -> bool:
    pgid = int(worker["proc"].pid)
    worker_id = str(worker["worker_id"])
    for sig, grace_sec in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        except Exception as exc:
            LOGGER.warning("Failed to signal worker process group %s pgid=%s: %s", worker_id, pgid, exc)
            return False
        if _wait_for_process_group_exit(pgid, timeout_sec=grace_sec):
            return True
    LOGGER.warning("Worker process group %s pgid=%s did not exit cleanly", worker_id, pgid)
    return False


def _launch_worker_process(cmd: List[str], *, fh, env: Dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )  # nosec: controlled args


def _parse_csv_table(text: str, columns: List[str]) -> List[Dict[str, str]]:
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


def _collect_gpu_snapshot(*, visible_devices: List[int]) -> Dict[str, Any]:
    gpu_text = _run_text_command(
        [
            "nvidia-smi",
            f"--id={','.join(str(device) for device in visible_devices)}",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,persistence_mode",
            "--format=csv,noheader,nounits",
        ]
    )
    process_text = _run_text_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "captured_at": _utc_now_iso(),
        "gpus": _parse_csv_table(
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
        "processes": _parse_csv_table(
            process_text,
            [
                "gpu_uuid",
                "pid",
                "process_name",
                "used_memory_mib",
            ],
        ),
    }


def _read_worker_runtime(runtime_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(runtime_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_runtime_summary(*, runtime_dir: Path, db_path: Path) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    first_batch_started = []
    last_batch_finished = []
    engine_ready = []
    for path in sorted(runtime_dir.glob("worker_*.runtime.json")):
        data = _read_worker_runtime(path)
        workers.append(data)
        first_batch_started_ts = _parse_utc_iso(data.get("first_batch_started_at"))
        last_batch_finished_ts = _parse_utc_iso(data.get("last_batch_finished_at"))
        engine_ready_ts = _parse_utc_iso(data.get("engine_ready_at"))
        if first_batch_started_ts is not None:
            first_batch_started.append(first_batch_started_ts)
        if last_batch_finished_ts is not None:
            last_batch_finished.append(last_batch_finished_ts)
        if engine_ready_ts is not None:
            engine_ready.append(engine_ready_ts)
    steady_summary = {
        "first_batch_started_at": _utc_now_iso(min(first_batch_started)) if first_batch_started else None,
        "last_batch_finished_at": _utc_now_iso(max(last_batch_finished)) if last_batch_finished else None,
        "all_workers_ready_at": _utc_now_iso(max(engine_ready)) if engine_ready else None,
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
                "generated_at": _utc_now_iso(),
                "queue_counts": work_queue_counts(db_path),
                "work_items": list(iter_work_items(db_path)),
                "workers": workers,
                "steady_state": steady_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def _query_persistence_mode(*, visible_devices: List[int]) -> List[Dict[str, str]]:
    raw = _run_text_command(
        [
            "nvidia-smi",
            f"--id={','.join(str(device) for device in visible_devices)}",
            "--query-gpu=index,persistence_mode",
            "--format=csv,noheader,nounits",
        ]
    )
    return _parse_csv_table(raw, ["index", "persistence_mode"])


def _ensure_gpu_preflight(*, visible_devices: List[int], mode: str) -> Dict[str, Any]:
    mode_norm = str(mode or "warn").strip().lower()
    status = {
        "mode": mode_norm,
        "checked_at": _utc_now_iso(),
        "before": _query_persistence_mode(visible_devices=visible_devices),
        "changed": False,
    }
    disabled = [item for item in status["before"] if str(item.get("persistence_mode", "")).lower() != "enabled"]
    if not disabled or mode_norm == "off":
        status["after"] = list(status["before"])
        return status
    if mode_norm == "ensure":
        try:
            subprocess.run(["sudo", "-n", "nvidia-smi", "-pm", "1"], check=True, capture_output=True, text=True)  # nosec: controlled args
            status["changed"] = True
        except Exception as exc:
            status["ensure_error"] = str(exc)
    status["after"] = _query_persistence_mode(visible_devices=visible_devices)
    return status


def _collect_xid_faults(*, start_utc_iso: str) -> Dict[str, Any]:
    cmd = [
        "journalctl",
        "-k",
        "--since",
        str(start_utc_iso),
        "--no-pager",
    ]
    try:
        output = _run_text_command(cmd)
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


def _start_gpu_telemetry(
    *,
    telemetry_path: Path,
    visible_devices: List[int],
    interval_sec: float,
    stop_event: threading.Event,
) -> threading.Thread:
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    def _loop() -> None:
        while not stop_event.wait(float(max(1.0, interval_sec))):
            try:
                with telemetry_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"kind": "sample", **_collect_gpu_snapshot(visible_devices=visible_devices)}) + "\n")
            except Exception as exc:  # pragma: no cover - best effort logging
                LOGGER.warning("GPU telemetry sample failed: %s", exc)

    thread = threading.Thread(target=_loop, name="deepseek-gpu-telemetry", daemon=True)
    thread.start()
    return thread


def _parse_shard_stem(stem: str) -> Optional[Dict[str, Any]]:
    match = SHARD_STEM_RE.match(str(stem))
    if match is None:
        return None
    return {
        "source_stem": str(match.group("source_stem")),
        "start_page": int(match.group("start")),
        "end_page": int(match.group("end")),
    }


def _split_markdown_pages(markdown_text: str, *, expected_pages: int) -> List[str]:
    pages = _split_page_outputs(markdown_text)
    if len(pages) < int(expected_pages):
        pages.extend([""] * (int(expected_pages) - len(pages)))
    elif len(pages) > int(expected_pages):
        pages = pages[: int(expected_pages)]
    return pages


def _archive_shard_artifact(*, out_root: Path, source_path: Path, relative_path: Path) -> None:
    archive_path = out_root / "sidecars" / "ocr_shards" / relative_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    source_path.replace(archive_path)


def _reassemble_canonical_output_for_source(
    *,
    out_root: Path,
    pdf_path: Path,
    source_name: str,
) -> bool:
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    source_stem = Path(source_name).stem
    canonical_md = md_dir / f"{source_stem}.md"
    canonical_metrics = metrics_dir / f"{source_stem}.metrics.json"
    if canonical_md.exists() and canonical_metrics.exists():
        return True

    shard_records: List[Dict[str, Any]] = []
    for metrics_path in sorted(metrics_dir.glob(f"{source_stem}__p*.metrics.json")):
        shard_stem = metrics_path.name.removesuffix(".metrics.json")
        shard_md = md_dir / f"{shard_stem}.md"
        if not shard_md.exists():
            continue
        shard_meta = _parse_shard_stem(shard_stem)
        if shard_meta is None:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        start_page = int(metrics.get("source_start_page", shard_meta["start_page"]))
        end_page = int(metrics.get("source_end_page", shard_meta["end_page"]))
        shard_records.append(
            {
                "stem": shard_stem,
                "md_path": shard_md,
                "metrics_path": metrics_path,
                "metrics": metrics,
                "start_page": start_page,
                "end_page": end_page,
            }
        )

    if not shard_records:
        return False

    shard_records.sort(key=lambda item: (int(item["start_page"]), int(item["end_page"]), str(item["stem"])))
    page_count = max(int(_page_count(pdf_path)), max(int(item["end_page"]) for item in shard_records))
    merged_pages = [""] * int(page_count)
    merged_page_metrics: List[Optional[Dict[str, Any]]] = [None] * int(page_count)
    merged_extra_metrics: Dict[str, Any] = {}
    repair_totals: Dict[str, int] = {}
    render_sec_total = 0.0
    infer_sec_total = 0.0
    wall_time_sec_total = 0.0
    reassembled_ranges: List[Dict[str, int]] = []

    for shard in shard_records:
        metrics = dict(shard["metrics"])
        start_page = int(shard["start_page"])
        end_page = int(shard["end_page"])
        expected_pages = max(0, end_page - start_page + 1)
        reassembled_ranges.append({"start_page": start_page, "end_page": end_page})

        shard_pages = _split_markdown_pages(
            shard["md_path"].read_text(encoding="utf-8"),
            expected_pages=expected_pages,
        )
        for offset, page_text in enumerate(shard_pages):
            merged_pages[start_page - 1 + offset] = page_text

        for idx, page_metric in enumerate(list(metrics.get("page_metrics") or []), start=1):
            absolute_page = start_page + int(page_metric.get("page_number", idx)) - 1
            if absolute_page <= 0 or absolute_page > int(page_count):
                continue
            merged_metric = dict(page_metric)
            merged_metric["page_number"] = int(absolute_page)
            merged_page_metrics[absolute_page - 1] = merged_metric

        render_sec_total += float(metrics.get("render_sec", 0.0))
        infer_sec_total += float(metrics.get("infer_sec_total", 0.0))
        wall_time_sec_total += float(metrics.get("wall_time_sec", 0.0))
        for key, value in dict(metrics.get("repair_summary") or {}).items():
            if key == "repair_mode":
                continue
            repair_totals[key] = int(repair_totals.get(key, 0)) + int(value)
        for key in REASSEMBLED_CONFIG_KEYS:
            if key in metrics and key not in merged_extra_metrics:
                merged_extra_metrics[key] = metrics[key]

    merged_extra_metrics.update(
        {
            "source_file": str(source_name),
            "source_stem": str(source_stem),
            "source_start_page": 1,
            "source_end_page": int(page_count),
            "reassembled_from_shards": True,
            "reassembled_shard_count": len(shard_records),
            "reassembled_source_ranges": reassembled_ranges,
            "render_sec": float(render_sec_total),
            "infer_sec_total": float(infer_sec_total),
            "wall_time_sec": float(wall_time_sec_total),
            "wall_time_sec_semantics": "sum_of_shard_wall_times",
            "page_metrics": [item for item in merged_page_metrics if item is not None],
        }
    )
    if repair_totals:
        merged_extra_metrics["repair_summary"] = {
            "repair_mode": str(merged_extra_metrics.get("repair_mode", "unknown")),
            **{key: int(value) for key, value in repair_totals.items()},
        }

    merged_markdown = _join_page_outputs(merged_pages) if merged_pages else "[[Blank page]]"
    _write_outputs(
        output_dir=out_root,
        stem=source_stem,
        markdown=merged_markdown,
        page_count=int(page_count),
        extra_metrics=merged_extra_metrics,
    )
    for shard in shard_records:
        _archive_shard_artifact(
            out_root=out_root,
            source_path=Path(shard["md_path"]),
            relative_path=Path("markdown") / Path(shard["md_path"]).name,
        )
        _archive_shard_artifact(
            out_root=out_root,
            source_path=Path(shard["metrics_path"]),
            relative_path=Path("json") / "metrics" / Path(shard["metrics_path"]).name,
        )
    return True


def _ensure_canonical_outputs(*, out_root: Path, pdf_root: Path, file_list: List[str]) -> None:
    for name in file_list:
        pdf_path = (pdf_root / name).resolve()
        if _reassemble_canonical_output_for_source(
            out_root=out_root,
            pdf_path=pdf_path,
            source_name=name,
        ):
            continue



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
    runtime_backend: str = "transformers",
    ocr_profile: str = "markdown_grounded",
    prompt_override: Optional[str] = None,
    attn_backend: str = "auto",
    base_size: Optional[int] = None,
    image_size: Optional[int] = None,
    crop_mode: Optional[bool] = None,
    render_dpi: Optional[int] = None,
    max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS,
    repetition_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    use_gpus: Optional[str] = None,
    devices: Optional[List[int]] = None,
    workers_per_gpu: int = 1,
    gpu_memory_utilization: Optional[float] = None,
    disable_fp8_kv: bool = False,
    vllm_batch_size: Optional[int] = None,
    repair_mode: str = "auto",
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
        runtime_backend or os.environ.get("GLOSSAPI_DEEPSEEK_RUNTIME_BACKEND", "transformers")
    ).strip().lower()
    if runtime_backend_norm not in {"transformers", "vllm"}:
        raise ValueError("runtime_backend must be 'transformers' or 'vllm'")

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
            render_dpi=render_dpi,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            runtime_backend=runtime_backend_norm,
            vllm_batch_size=vllm_batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_fp8_kv=disable_fp8_kv,
            repair_mode=repair_mode,
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
            render_dpi=render_dpi,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            runtime_backend=runtime_backend_norm,
            vllm_batch_size=resolved_vllm_batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_fp8_kv=disable_fp8_kv,
            repair_mode=repair_mode,
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
        text_payload = md_path.read_text(encoding="utf-8")
        page_count = _page_count(pdf_path)
        result_payload: Optional[Dict[str, Any]] = None
        if metrics_path.exists():
            try:
                result_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                result_payload = None
        if result_payload is None:
            result_payload = {"page_count": page_count}
        else:
            result_payload.setdefault("page_count", page_count)
        if not text_payload.strip():
            result_payload["empty_markdown"] = True
            metrics_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            LOGGER.warning("DeepSeek OCR produced empty markdown for %s: %s", name, md_path)
            results[stem] = result_payload
            continue
        results[stem] = result_payload
        if not metrics_path.exists():
            metrics_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    return results
