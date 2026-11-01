"""Disk the final products need on the corpus filesystem (K4(b) round 1: K4BI-G7, G8, G10).

The document export and the annotated-section export are the user's output. Under memory_saving=extreme they are
charged to the corpus filesystem, never to the per-attempt staging cap, and a run whose corpus filesystem cannot hold
them is refused with a typed error at two points that come before any publication: at run start (estimated from the
sources' sizes and a declared planning ratio) and after the document stages (from the exact text sizes the run
produced). The text the corpus barriers read stays until the publications (G8, inherent: the final products are made
from it); ``plan --json`` shows both estimates.

Factors are bytes per byte of cleaned text, measured on the K4(b) Mac corpus (8 documents, 1.25 MB of cleaned text):
document export 1.02, sections parquet 0.88, classified and annotated parquets 0.87 each, annotation partitions 0.98,
section fragments 2.2. A final product is kept twice (its generation and the published copy), and an aggregate stage
holds a transient copy of its inputs; the factors below round those up.
"""
from __future__ import annotations

import os
from pathlib import Path

PLANNING_TEXT_RATIO = 0.1     # cleaned text per source byte assumed before any text exists (cohort242 ~0.06, Mac 0.08)
PER_DOCUMENT_BYTES = 8192     # export record metadata per document, rounded up
PRODUCT_FACTORS = {'export': 3.2, 'sections': 2.9, 'annotate': 5.3}
KEPT_TEXT_FACTORS = {'extract': 1.05, 'clean': 1.0, 'sections': 2.2}  # raw markdown, cleaned text, section fragments


def product_need(stages, text_bytes, documents):
    """Bytes the publications add to the corpus filesystem for this much cleaned text."""
    factor = sum(value for stage, value in PRODUCT_FACTORS.items() if stage in stages)
    return int(factor*text_bytes + (PER_DOCUMENT_BYTES*documents if 'export' in stages else 0))


def kept_text(stages, text_bytes):
    """Bytes of text intermediates the corpus barriers keep until the publications (K4BI-G8)."""
    return int(sum(value for stage, value in KEPT_TEXT_FACTORS.items() if stage in stages)*text_bytes)


def estimate(stages, source_bytes, documents, *, unknown=0, known_text=None):
    """The ``plan --json`` estimate; ``known_text`` replaces the planning ratio when the ledger knows the text."""
    text = known_text if known_text is not None else int(PLANNING_TEXT_RATIO*source_bytes)
    return dict(text_bytes=text, kept_text_bytes=kept_text(stages, text), final_product_bytes=product_need(stages, text, documents),
                planning_text_ratio=PLANNING_TEXT_RATIO, sources_without_size=unknown,
                basis='cleaned text = planning ratio x source bytes (the ledger\'s recorded text when known); final products and '
                      'kept text scale with it (stages.final_products factors). Under memory_saving=extreme the run refuses at start '
                      'and again, from exact text sizes, before the publications when the corpus filesystem cannot hold them.')


def corpus_room(root, settings):
    """Corpus-filesystem bytes available to products: free space minus the emergency reserve and one attempt reservation."""
    from .disk_budget import attempt_reservation, emergency_reserve
    state = os.statvfs(Path(root))
    return state.f_bavail*state.f_frsize - emergency_reserve(settings.get('disk_budget_bytes')) - attempt_reservation(settings)


def require_room(root, settings, need, *, when, present=None):
    """Typed refusal (``disk_budget_exceeded``) when the final products cannot fit; extreme only."""
    if settings.get('memory_saving') != 'extreme':
        return
    from .disk_budget import DiskBudgetError
    room = corpus_room(root, settings)
    limit = settings.get('disk_budget_bytes')
    if need > room or (limit is not None and present is not None and present+need > limit):
        raise DiskBudgetError('disk_budget_exceeded', f'the final products need about {need} bytes on the corpus filesystem {when}; '
                              f'{room} are available after the emergency reserve and one attempt reservation'
                              + (f', and the budget leaves {limit-present}' if limit is not None and present is not None else '')
                              + '. Enlarge the corpus filesystem (and disk_budget_bytes) or request fewer products')
