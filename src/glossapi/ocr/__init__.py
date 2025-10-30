"""Lightweight OCR backend package.

Exports minimal, import-safe helpers for OCR backends. Heavy
dependencies (vLLM, transformers, PyMuPDF) are imported lazily
inside the specific backend functions so importing this package
does not require GPU stacks or model weights.
"""

from __future__ import annotations

import importlib

__all__ = [
    "deepseek",
    "rapidocr",
    "math",
    "utils",
    "deepseek_runner",
    "rapidocr_dispatch",
]

_SUBPACKAGES = {"deepseek", "rapidocr", "math", "utils"}
_ALIASES = {
    "deepseek_runner": "glossapi.ocr.deepseek.runner",
    "rapidocr_dispatch": "glossapi.ocr.rapidocr.dispatch",
}


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"glossapi.ocr.{name}")
    target = _ALIASES.get(name)
    if target:
        return importlib.import_module(target)
    raise AttributeError(name)
