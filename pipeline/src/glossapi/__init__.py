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

from .gloss_extract import GlossExtract
from .gloss_section_classifier import GlossSectionClassifier
from .corpus import Corpus
from .sampler import Sampler
from .gloss_section import Section, GlossSection
from .gloss_downloader import GlossDownloader

__all__ = [
    'GlossExtract',
    'GlossSection',
    'GlossSectionClassifier',
    'Corpus',
    'Sampler',
    'Section',
    'NewGlossSection',
    'GlossDownloader'
]

__version__ = '0.0.10'
