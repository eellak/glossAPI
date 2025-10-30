"""
GlossAPI Library

A library for processing academic texts in Greek and other languages:
- Extracting content from PDFs and other formats with Docling
- Robust batch processing with error isolation and automatic resumption
- Clustering documents based on extraction quality
- Extracting and cleaning academic sections
- Classifying sections using machine learning

This is an open source project that provides tools for linguistic annotations
and text processing, with a special focus on the Greek language.
"""

from __future__ import annotations

import os

# Keep Docling/RapidOCR bootstrap optional and import‑light by default.
# If the environment requests skipping (common in tests or minimal envs),
# or if Docling is not installed, we avoid importing heavy dependencies here.
_SKIP_DOCLING_BOOT = os.environ.get("GLOSSAPI_SKIP_DOCLING_BOOT") == "1"

def _attempt_patch_docling() -> bool:
    if _SKIP_DOCLING_BOOT:
        return False
    try:
        # Import inside the function to avoid pulling Docling when unused or missing.
        from .ocr.rapidocr.safe import patch_docling_rapidocr  # type: ignore

        try:
            return bool(patch_docling_rapidocr())
        except Exception:
            # Swallow any runtime error to keep top‑level import light/safe.
            return False
    except Exception:
        # Docling (or its transitive deps) not available – keep going.
        return False


def patch_docling_rapidocr() -> bool:
    """Best‑effort registration of the SafeRapidOcrModel.

    Returns True when the patch was applied; False when unavailable or skipped.
    Safe to call multiple times.
    """
    return _attempt_patch_docling()

# Attempt the patch once at import time, but never fail import if it does not apply.
_ = _attempt_patch_docling()

__all__ = [
    'GlossSection',
    'GlossSectionClassifier',
    'Corpus',
    'Sampler',
    'Section',
    'GlossDownloader',
    'patch_docling_rapidocr',
]

def __getattr__(name: str):
    # Lazy access for heavy modules to keep top‑level import light.
    if name == 'Corpus':
        from .corpus.corpus_orchestrator import Corpus  # type: ignore
        return Corpus
    if name == 'GlossSectionClassifier':
        from .gloss_section_classifier import GlossSectionClassifier  # type: ignore
        return GlossSectionClassifier
    if name == 'Sampler':
        from .sampler import Sampler  # type: ignore
        return Sampler
    if name == 'GlossSection':
        from .gloss_section import GlossSection  # type: ignore
        return GlossSection
    if name == 'Section':
        from .gloss_section import Section  # type: ignore
        return Section
    if name == 'GlossDownloader':
        from .gloss_downloader import GlossDownloader  # type: ignore
        return GlossDownloader
    raise AttributeError(name)

# Derive version dynamically from installed package metadata if possible
try:
    from importlib.metadata import version as _pkg_version
    __version__: str = _pkg_version(__name__)
except Exception:
    __version__ = "0.1.1"
