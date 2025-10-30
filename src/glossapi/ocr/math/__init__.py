"""Math enrichment subpackage with lazy accessors."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "enrich",
    "enrich_from_docling_json",
    "RoiEntry",
    "PageRasterCache",
    "earlystop",
]


def _load() -> Any:
    return import_module("glossapi.ocr.math.enrich")


def __getattr__(name: str) -> Any:
    if name == "earlystop":
        return import_module("glossapi.ocr.math.earlystop")
    mod = _load()
    if name == "enrich":
        return mod
    if hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
