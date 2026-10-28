"""Compatibility wrapper for the canonical Docling pipeline builder."""

from glossapi.optional_dependencies import optional_dependencies

with optional_dependencies("layout"):
    from .docling.pipeline import build_layout_pipeline

__all__ = ["build_layout_pipeline"]
