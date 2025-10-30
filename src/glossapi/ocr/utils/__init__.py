"""OCR utilities subpackage with lazy helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "cleaning",
    "json_io",
    "page",
    "triage",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"glossapi.ocr.utils.{name}")
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
