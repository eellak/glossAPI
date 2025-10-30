"""RapidOCR subpackage with lazy re-exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "dispatch",
    "docling_pipeline",
    "pool",
    "safe",
    "onnx",
    "_paths",
    "pipeline",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"glossapi.ocr.rapidocr.{name}")
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
