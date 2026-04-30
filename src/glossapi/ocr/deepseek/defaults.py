"""Canonical DeepSeek OCR defaults shared across orchestration and CLIs."""

from __future__ import annotations

from typing import Optional

DEFAULT_RUNTIME_BACKEND = "vllm"
DEFAULT_OCR_PROFILE = "markdown_grounded"
DEFAULT_ATTN_BACKEND = "auto"
DEFAULT_RENDER_DPI = 144
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_REPAIR_MODE = "auto"
DEFAULT_WORKERS_PER_GPU = 1
DEFAULT_TARGET_BATCH_PAGES = 160


def resolve_render_dpi(value: Optional[int]) -> int:
    """Return the canonical render DPI, even when callers pass ``None``."""

    return DEFAULT_RENDER_DPI if value is None else int(value)


def resolve_gpu_memory_utilization(value: Optional[float]) -> float:
    """Return the canonical vLLM memory target, even when callers pass ``None``."""

    return DEFAULT_GPU_MEMORY_UTILIZATION if value is None else float(value)
