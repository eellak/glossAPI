"""
GlossAPI Library

A library for processing academic texts in Greek and other languages:
- Extracting content from PDFs with Docling
- Clustering documents based on extraction quality
- Extracting and cleaning academic sections
- Classifying sections using machine learning

This is an open source project that provides tools for linguistic annotations
and text processing, with a special focus on the Greek language.
"""

from .academic_section import AcademicSection
from .gloss_convert import GlossConvert
from .gloss_extraction import GlossExtraction
from .gloss_academic_classifier import GlossAcademicClassifier
from .corpus import Corpus, Sampler

__all__ = [
    'AcademicSection',
    'GlossConvert',
    'GlossExtraction',
    'GlossAcademicClassifier',
    'Corpus',
    'Sampler'
]

__version__ = '0.0.3'