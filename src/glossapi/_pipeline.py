"""Backward-compatible adapter.

Docling pipeline builders moved to `glossapi.ocr.docling.pipeline`.
This module re-exports the public API to preserve legacy imports.
"""

from .ocr.docling.pipeline import *  # noqa: F401,F403
