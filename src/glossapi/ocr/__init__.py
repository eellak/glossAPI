"""Lightweight OCR backend package.

Exports minimal, import-safe helpers for OCR backends. Heavy
dependencies (transformers, PyMuPDF) are imported lazily
inside the specific backend functions so importing this package
does not require GPU stacks or model weights.
"""

from __future__ import annotations

import importlib

__all__ = [
    "deepseek",
    "math",
    "utils",
    "deepseek_runner",
]

_SUBPACKAGES = {"deepseek", "math", "utils"}
_ALIASES = {
    "deepseek_runner": "glossapi.ocr.deepseek.runner",
}


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"glossapi.ocr.{name}")
    target = _ALIASES.get(name)
    if target:
        return importlib.import_module(target)
    raise AttributeError(name)
