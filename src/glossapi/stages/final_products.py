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
from typing import Any, Iterable

PLANNING_TEXT_RATIO = 0.1     # ESTIMATE only: cleaned text per source byte assumed before any text exists (cohort242 ~0.06, Mac 0.08)
PER_DOCUMENT_BYTES = 8192     # ESTIMATE only (run start, plan --json): export record metadata per document
RECORD_OVERHEAD_BYTES = 4096  # binding check: the fixed schema's keys with null values plus computed fields (doc_id, counters, flags); measured 1.6 KB per record, rounded up. Producer metadata is counted exactly, not here
EXPORT_TEXT_FACTOR = 3.2      # the export's share of PRODUCT_FACTORS: replaced by exact sizes in the binding check
PRODUCT_FACTORS = {'export': 3.2, 'sections': 2.9, 'annotate': 5.3}
KEPT_TEXT_FACTORS = {'extract': 1.05, 'clean': 1.0, 'sections': 2.2}  # raw markdown, cleaned text, section fragments


def product_need(stages: Iterable[str], text_bytes: int, documents: int) -> int:
    """Bytes the publications add to the corpus filesystem for this much cleaned text."""
    requested = set(stages)
    factor = sum(value for stage, value in PRODUCT_FACTORS.items() if stage in requested)
    return int(factor*text_bytes + (PER_DOCUMENT_BYTES*documents if 'export' in requested else 0))


def exact_export_need(texts: Iterable[Any], metadata_paths: Iterable[Any], stems: set[str] | None) -> int:
    """Binding pre-publication bound of the document export, from actual sizes (CP-K4a F7).

    Each record is its text and its metadata row serialized as the export serializes them (JSON, ``ensure_ascii=False``,
    escapes included) plus ``RECORD_OVERHEAD_BYTES`` of computed fields. The export holds the records twice at once
    (its generation and the published copy) and, while it runs, a transient view of its inputs (the cleaned texts and
    the metadata table). ``texts``: paths of this run's cleaned texts; ``metadata_paths``: the metadata table and the
    run's source projection (the caller's manifest columns that the download publication merges into the row; their
    sizes are added, an upper bound of the merged row); ``stems``: the planned compatibility stems."""
    import importlib
    import json
    import re
    from .._naming import canonical_stem
    record_bytes = view_bytes = 0
    for path in texts:
        text = Path(path).read_text(encoding='utf-8', errors='replace')
        record_bytes += len(json.dumps(text, ensure_ascii=False).encode()) + RECORD_OVERHEAD_BYTES
        view_bytes += Path(path).stat().st_size
    pd: Any = importlib.import_module("pandas")
    for index, metadata_path in enumerate(metadata_paths):
        if metadata_path is None or not Path(metadata_path).is_file():
            continue
        frame: Any = pd.read_parquet(metadata_path)
        if index == 0:
            view_bytes += Path(metadata_path).stat().st_size
        names: Any = frame['filename'] if 'filename' in frame.columns else frame['filename_base'] if 'filename_base' in frame.columns else None
        if names is None:
            continue
        rows: list[dict[str, Any]] = frame.to_dict(orient='records')
        for name, row in zip(list(names), rows):
            stem = re.sub(r'__p\d{4}-\d{4}$', '', canonical_stem(str(name or '')))
            if stems is None or stem in stems:
                record_bytes += len(json.dumps(row, ensure_ascii=False, default=str).encode())
    return 2*record_bytes + view_bytes


def binding_need(stages: Iterable[str], text_bytes: int, documents: int, export_bytes: int) -> int:
    """``product_need`` with the export's text factor and per-document allowance replaced by ``exact_export_need``."""
    requested = set(stages)
    other = sum(value for stage, value in PRODUCT_FACTORS.items() if stage in requested and stage != 'export')
    return int(other*text_bytes + (export_bytes if 'export' in requested else 0))


def kept_text(stages: Iterable[str], text_bytes: int) -> int:
    """Bytes of text intermediates the corpus barriers keep until the publications (K4BI-G8)."""
    requested = set(stages)
    return int(sum(value for stage, value in KEPT_TEXT_FACTORS.items() if stage in requested)*text_bytes)


def estimate(stages: Iterable[str], source_bytes: int, documents: int, *, unknown: int = 0, known_text: int | None = None) -> dict[str, Any]:
    """The ``plan --json`` estimate; ``known_text`` replaces the planning ratio when the ledger knows the text."""
    stages = list(stages)
    text = known_text if known_text is not None else int(PLANNING_TEXT_RATIO*source_bytes)
    return dict(text_bytes=text, kept_text_bytes=kept_text(stages, text), final_product_bytes=product_need(stages, text, documents),
                planning_text_ratio=PLANNING_TEXT_RATIO, sources_without_size=unknown,
                basis='cleaned text = planning ratio x source bytes (the ledger\'s recorded text when known); final products and '
                      'kept text scale with it (stages.final_products factors). Under memory_saving=extreme the run refuses at start '
                      'and again, from exact text sizes, before the publications when the corpus filesystem cannot hold them.')


def corpus_room(root: Any, settings: dict[str, Any]) -> int:
    """Corpus-filesystem bytes available to products: free space minus the emergency reserve and one attempt reservation."""
    from .disk_budget import attempt_reservation, emergency_reserve
    state = os.statvfs(Path(root))
    return state.f_bavail*state.f_frsize - emergency_reserve(settings.get('disk_budget_bytes')) - attempt_reservation(settings)


def require_room(root: Any, settings: dict[str, Any], need: int, *, when: str, present: int | None = None) -> None:
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
