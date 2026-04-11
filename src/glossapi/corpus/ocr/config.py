"""Request normalization for corpus OCR orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...ocr.deepseek.defaults import (
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


@dataclass(slots=True)
class OcrRequest:
    mode: str
    backend: str
    device: Optional[str]
    model_dir: Optional[Path]
    max_pages: Optional[int]
    persist_engine: bool
    precision: Optional[str]
    workers_per_gpu: int
    runtime_backend: str
    ocr_profile: str
    prompt_override: Optional[str]
    attn_backend: str
    base_size: Optional[int]
    image_size: Optional[int]
    crop_mode: Optional[bool]
    render_dpi: int
    max_new_tokens: int
    repetition_penalty: Optional[float]
    no_repeat_ngram_size: Optional[int]
    vllm_batch_size: Optional[int]
    gpu_memory_utilization: float
    disable_fp8_kv: bool
    repair_mode: str
    repair_exec_batch_target_pages: Optional[int]
    repair_exec_batch_target_items: Optional[int]
    scheduler: str
    target_batch_pages: int
    shard_pages: int
    shard_threshold_pages: int
    math_enhance: bool
    math_targets: Optional[Dict[str, List[Tuple[int, int]]]]
    math_batch_size: int
    math_dpi_base: int
    use_gpus: str
    devices: Optional[List[int]]
    reprocess_completed: bool
    content_debug: bool


def _resolve_mode(
    *,
    logger,
    mode: Optional[str],
    fix_bad: bool,
    math_enhance: bool,
) -> Optional[str]:
    mode_norm: Optional[str] = None
    if mode:
        candidate = str(mode).strip().lower()
        if candidate in {"ocr_bad", "math_only", "ocr_bad_then_math"}:
            mode_norm = candidate
        else:
            logger.warning("Unknown mode '%s'; falling back to legacy flags", mode)
    if mode_norm is None:
        if fix_bad and math_enhance:
            mode_norm = "ocr_bad_then_math"
        elif fix_bad:
            mode_norm = "ocr_bad"
        elif math_enhance:
            mode_norm = "math_only"
    return mode_norm


def normalize_ocr_request(
    *,
    logger,
    fix_bad: bool,
    mode: Optional[str],
    backend: str,
    device: Optional[str],
    model_dir: Optional[str | Path],
    max_pages: Optional[int],
    persist_engine: bool,
    precision: Optional[str],
    workers_per_gpu: int = DEFAULT_WORKERS_PER_GPU,
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
    vllm_batch_size: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = DEFAULT_GPU_MEMORY_UTILIZATION,
    disable_fp8_kv: bool = False,
    repair_mode: str = DEFAULT_REPAIR_MODE,
    repair_exec_batch_target_pages: Optional[int] = None,
    repair_exec_batch_target_items: Optional[int] = None,
    scheduler: str = "auto",
    target_batch_pages: int = DEFAULT_TARGET_BATCH_PAGES,
    shard_pages: int = 0,
    shard_threshold_pages: int = 0,
    math_enhance: bool = True,
    math_targets: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    math_batch_size: int = 8,
    math_dpi_base: int = 220,
    use_gpus: str = "single",
    devices: Optional[List[int]] = None,
    force: Optional[bool] = None,
    reprocess_completed: Optional[bool] = None,
    skip_existing: Optional[bool] = None,
    content_debug: bool = False,
    CONTENT_DEBUG: Optional[bool] = None,
    internal_debug: bool = False,
    INTERNAL_DEBUG: Optional[bool] = None,
) -> Optional[OcrRequest]:
    backend_norm = str(backend or "deepseek").strip().lower()
    if backend_norm != "deepseek":
        raise ValueError("backend must be 'deepseek'")

    if CONTENT_DEBUG is not None:
        content_debug = bool(CONTENT_DEBUG)
    elif INTERNAL_DEBUG is not None:
        content_debug = bool(INTERNAL_DEBUG)
    elif internal_debug:
        content_debug = True

    fix_bad_effective = bool(fix_bad)
    if force is not None:
        logger.warning("Corpus.ocr(force=...) is deprecated; use fix_bad=... instead")
        fix_bad_effective = bool(force)

    mode_norm = _resolve_mode(
        logger=logger,
        mode=mode,
        fix_bad=fix_bad_effective,
        math_enhance=bool(math_enhance),
    )
    if mode_norm is None:
        logger.info(
            "OCR: no operation requested (enable fix_bad and/or math_enhance or set mode='ocr_bad'|'math_only'|'ocr_bad_then_math')"
        )
        return None

    reprocess_explicit = reprocess_completed is not None
    reprocess_flag = bool(reprocess_completed) if reprocess_explicit else False
    if skip_existing is not None:
        skip_flag = bool(skip_existing)
        logger.warning(
            "Corpus.ocr(skip_existing=...) is deprecated; use reprocess_completed=... instead."
        )
        desired = not skip_flag
        if reprocess_explicit and desired != reprocess_flag:
            logger.info(
                "Corpus.ocr(): skip_existing=%s overrides reprocess_completed=%s (effective reprocess_completed=%s).",
                skip_flag,
                reprocess_flag,
                desired,
            )
        reprocess_flag = desired

    return OcrRequest(
        mode=mode_norm,
        backend=backend_norm,
        device=device,
        model_dir=Path(model_dir) if model_dir else None,
        max_pages=max_pages,
        persist_engine=bool(persist_engine),
        precision=precision,
        workers_per_gpu=int(max(1, workers_per_gpu)),
        runtime_backend=str(runtime_backend or DEFAULT_RUNTIME_BACKEND),
        ocr_profile=str(ocr_profile or DEFAULT_OCR_PROFILE),
        prompt_override=prompt_override,
        attn_backend=str(attn_backend or DEFAULT_ATTN_BACKEND),
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        render_dpi=resolve_render_dpi(render_dpi),
        max_new_tokens=int(DEFAULT_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens),
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        vllm_batch_size=vllm_batch_size,
        gpu_memory_utilization=resolve_gpu_memory_utilization(gpu_memory_utilization),
        disable_fp8_kv=bool(disable_fp8_kv),
        repair_mode=str(repair_mode or DEFAULT_REPAIR_MODE),
        repair_exec_batch_target_pages=repair_exec_batch_target_pages,
        repair_exec_batch_target_items=repair_exec_batch_target_items,
        scheduler=str(scheduler or "auto"),
        target_batch_pages=int(max(1, target_batch_pages)),
        shard_pages=int(max(0, shard_pages)),
        shard_threshold_pages=int(max(0, shard_threshold_pages)),
        math_enhance=bool(math_enhance),
        math_targets=math_targets,
        math_batch_size=int(math_batch_size),
        math_dpi_base=int(math_dpi_base),
        use_gpus=str(use_gpus or "single"),
        devices=devices,
        reprocess_completed=bool(reprocess_flag),
        content_debug=bool(content_debug),
    )
