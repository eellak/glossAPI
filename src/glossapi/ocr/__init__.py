"""Lazy math and utility helpers for external Docling OCR engines."""
from __future__ import annotations
import importlib

__all__ = ["math", "utils", "page_artifacts"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"glossapi.ocr.{name}")
    raise AttributeError(name)
