"""Pure planning helpers for the DeepSeek runner shim."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from glossapi.ocr.deepseek.defaults import DEFAULT_TARGET_BATCH_PAGES
from glossapi.ocr.deepseek.scheduling import (
    SourceDocument,
    assign_batches_to_lanes,
    build_exact_fill_batches,
    build_fixed_shard_slices,
    build_whole_document_slices,
    pack_slices_into_batches,
)

try:
    import pypdfium2 as _pypdfium2
except Exception:  # pragma: no cover - optional dependency
    _pypdfium2 = None

AUTO_VLLM_BATCH_PAGE_CAP = DEFAULT_TARGET_BATCH_PAGES


def page_count(pdf_path: Path) -> int:
    if _pypdfium2 is None:
        return 0
    try:
        return len(_pypdfium2.PdfDocument(str(pdf_path)))
    except Exception:
        return 0


def parse_device_index(device: Optional[str]) -> Optional[int]:
    if not device:
        return None
    value = str(device).strip().lower()
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def detect_visible_gpus() -> List[int]:
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


def resolve_lane_devices(
    *,
    use_gpus: Optional[str],
    devices: Optional[List[int]],
    workers_per_gpu: int,
    device: Optional[str],
    detect_visible_gpus_fn: Callable[[], List[int]] = detect_visible_gpus,
    parse_device_index_fn: Callable[[Optional[str]], Optional[int]] = parse_device_index,
) -> List[int]:
    if devices:
        resolved = [int(dev) for dev in devices]
        if resolved:
            return resolved
    if str(use_gpus or "single").strip().lower() == "multi":
        resolved = detect_visible_gpus_fn()
        if resolved:
            return resolved
    if workers_per_gpu > 1:
        from_device = parse_device_index_fn(device)
        if from_device is not None:
            return [from_device]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            first = visible.split(",", 1)[0].strip()
            if first.isdigit():
                return [int(first)]
        return [0]
    return []


def effective_page_count(
    pdf_path: Path,
    max_pages: Optional[int],
    *,
    page_count_fn: Callable[[Path], int] = page_count,
) -> int:
    count = page_count_fn(pdf_path)
    if max_pages is not None and count > 0:
        return min(count, int(max_pages))
    return max(1, count)


def source_documents(
    *,
    file_list: List[str],
    input_root: Path,
    max_pages: Optional[int],
    page_count_fn: Callable[[Path], int] = page_count,
) -> List[SourceDocument]:
    documents: List[SourceDocument] = []
    for name in file_list:
        pdf_path = (input_root / name).resolve()
        documents.append(
            SourceDocument(
                name=str(name),
                pages=int(effective_page_count(pdf_path, max_pages, page_count_fn=page_count_fn)),
            )
        )
    return documents


def plan_lanes(
    *,
    file_list: List[str],
    input_root: Path,
    lane_devices: List[int],
    workers_per_gpu: int,
    max_pages: Optional[int],
    page_count_fn: Callable[[Path], int] = page_count,
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
        weighted_files.append(
            (name, effective_page_count(pdf_path, max_pages, page_count_fn=page_count_fn))
        )
    weighted_files.sort(key=lambda item: (-item[1], item[0]))

    for name, weight in weighted_files:
        lane = min(lanes, key=lambda item: int(item["weight"]))
        lane["files"].append(name)
        lane["weight"] = int(lane["weight"]) + int(weight)
    return lanes


def resolve_scheduler(
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


def plan_lane_batches(
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
    page_count_fn: Callable[[Path], int] = page_count,
) -> List[Dict[str, Any]]:
    documents = source_documents(
        file_list=file_list,
        input_root=input_root,
        max_pages=max_pages,
        page_count_fn=page_count_fn,
    )
    scheduler_norm = resolve_scheduler(
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


def plan_work_batches(
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
    page_count_fn: Callable[[Path], int] = page_count,
) -> List[Dict[str, Any]]:
    documents = source_documents(
        file_list=file_list,
        input_root=input_root,
        max_pages=max_pages,
        page_count_fn=page_count_fn,
    )
    scheduler_norm = resolve_scheduler(
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


def auto_vllm_batch_size(
    *,
    runtime_backend: str,
    file_list: List[str],
    input_root: Path,
    max_pages: Optional[int],
    page_count_fn: Callable[[Path], int] = page_count,
) -> Optional[int]:
    if str(runtime_backend or "").strip().lower() != "vllm":
        return None
    total_pages = 0
    for name in file_list:
        pdf_path = (input_root / name).resolve()
        total_pages += int(effective_page_count(pdf_path, max_pages, page_count_fn=page_count_fn))
    if total_pages <= 0:
        return 1
    return min(int(total_pages), int(AUTO_VLLM_BATCH_PAGE_CAP))


def auto_vllm_batch_size_for_pages(*, runtime_backend: str, pages: int) -> Optional[int]:
    if str(runtime_backend or "").strip().lower() != "vllm":
        return None
    if int(pages) <= 0:
        return 1
    return min(int(pages), int(AUTO_VLLM_BATCH_PAGE_CAP))


def flatten_lane_batches(lane: Dict[str, Any]) -> Dict[str, Any]:
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
