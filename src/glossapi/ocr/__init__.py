"""Lightweight OCR backend package.

Exports minimal, import-safe helpers for OCR backends. Heavy
dependencies (vLLM, transformers, PyMuPDF) are imported lazily
inside the specific backend functions so importing this package
does not require GPU stacks or model weights.
"""

from __future__ import annotations

# Re-export runner entry points without importing heavy stacks at module import

__all__ = ["deepseek_runner", "rapidocr_dispatch"]


def __getattr__(name: str):
    # Lazy import submodules when accessed as attributes
    if name in ("deepseek_runner", "rapidocr_dispatch"):
        import importlib

        return importlib.import_module(f"glossapi.ocr.{name}")
    raise AttributeError(name)
