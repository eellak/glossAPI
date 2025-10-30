"""Backward-compatible adapter.

Docling pipeline builders moved to `glossapi.ocr.rapidocr.pipeline`.
This module re-exports the public API to preserve legacy imports.
"""

from .ocr.rapidocr.pipeline import *  # noqa: F401,F403
