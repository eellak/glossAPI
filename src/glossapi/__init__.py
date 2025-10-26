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

_SKIP_DOCLING_BOOT = os.environ.get("GLOSSAPI_SKIP_DOCLING_BOOT") == "1"

if not _SKIP_DOCLING_BOOT:
    from .rapidocr_safe import patch_docling_rapidocr

    patch_docling_rapidocr()
else:
    def patch_docling_rapidocr() -> bool:
        """Placeholder when Docling bootstrap is skipped via env flag."""
        return False

from .gloss_section_classifier import GlossSectionClassifier
from .corpus import Corpus
from .sampler import Sampler
from .gloss_section import Section, GlossSection
from .gloss_downloader import GlossDownloader

__all__ = [
    'GlossSection',
    'GlossSectionClassifier',
    'Corpus',
    'Sampler',
    'Section',
    'NewGlossSection',
    'GlossDownloader'
]

# Derive version dynamically from installed package metadata if possible
try:
    from importlib.metadata import version as _pkg_version
    __version__: str = _pkg_version(__name__)
except Exception:
    __version__ = "0.1.1"
