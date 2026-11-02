"""Disk the final products need on the corpus filesystem (K4(b) round 1: K4BI-G7, G8, G10).

The document export and the annotated-section export are the user's output. Under memory_saving=extreme they are
charged to the corpus filesystem, never to the per-attempt staging cap, and a run whose corpus filesystem cannot hold
them is refused with a typed error before any publication writes: at run start (estimated from the sources' sizes and a
declared planning ratio), after the document stages, and again before each publication (CP-K4c3 R3: the publication
charge, ``publication_charges``, derived from the writers' own input and product lists). The text the corpus barriers
read stays until the publications (G8, inherent: the final products are made from it); ``plan --json`` shows the
run-start estimate.

Factors are bytes per byte of cleaned text, measured on the K4(b) Mac corpus (8 documents, 1.25 MB of cleaned text):
document export 1.02, sections parquet 0.88, classified and annotated parquets 0.87 each, annotation partitions 0.98,
section fragments 2.2. A final product is kept twice (its generation and the published copy), and an aggregate stage
holds a transient copy of its inputs; the factors below round those up. They serve only the run-start estimate.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

PLANNING_TEXT_RATIO = 0.1     # ESTIMATE only: cleaned text per source byte assumed before any text exists (cohort242 ~0.06, Mac 0.08)
PER_DOCUMENT_BYTES = 8192     # ESTIMATE only (run start, plan --json): export record metadata per document
RECORD_OVERHEAD_BYTES = 4096  # binding check: the fixed schema's keys with null values plus computed fields (doc_id, counters, flags); measured 1.6 KB per record, rounded up. Producer metadata is counted exactly, not here
EXPORT_TEXT_FACTOR = 3.2
# A compatibility metadata table that publication will create (a first run: it does not exist at the pre-publication
# check). Allowance for the table and, separately, the export's working copy of it: a fixed parquet footprint (one row
# measured 29,343 bytes: ~70 typed columns, schema, statistics) plus a per-document row allowance. Producer metadata
# (the caller's manifest columns) is counted exactly from the source projection. Bound checked against measured tables
# (CP-K4ac2 R3); the export barrier re-checks with the actual table before the export writes.
METADATA_TABLE_FIXED_BYTES = 65536
METADATA_TABLE_ROW_BYTES = 2048      # the export's share of PRODUCT_FACTORS: replaced by exact sizes in the binding check
PRODUCT_FACTORS = {'export': 3.2, 'sections': 2.9, 'annotate': 5.3}
KEPT_TEXT_FACTORS = {'extract': 1.05, 'clean': 1.0, 'sections': 2.2}  # raw markdown, cleaned text, section fragments


def product_need(stages: Iterable[str], text_bytes: int, documents: int) -> int:
    """Bytes the publications add to the corpus filesystem for this much cleaned text."""
    requested = set(stages)
    factor = sum(value for stage, value in PRODUCT_FACTORS.items() if stage in requested)
    return int(factor*text_bytes + (PER_DOCUMENT_BYTES*documents if 'export' in requested else 0))


def exact_export_need(texts: Iterable[Any], metadata_paths: Iterable[Any], stems: set[str] | None) -> int:
    """The round-1/2 export bound from actual sizes (CP-K4a F7). The binding check now charges every publication with
    ``publication_charges`` (CP-K4c3 R3); this record-and-view sum stays as a lower building block of it.

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
    record_bytes = view_bytes = documents = 0
    for path in texts:
        documents += 1
        text = Path(path).read_text(encoding='utf-8', errors='replace')
        record_bytes += len(json.dumps(text, ensure_ascii=False).encode()) + RECORD_OVERHEAD_BYTES
        view_bytes += Path(path).stat().st_size
    pd: Any = importlib.import_module("pandas")
    for index, metadata_path in enumerate(metadata_paths):
        if index == 0 and (metadata_path is None or not Path(metadata_path).is_file()):
            # Not created yet: publication creates the table and the export copies it into its view (both counted).
            view_bytes += 2*(METADATA_TABLE_FIXED_BYTES + METADATA_TABLE_ROW_BYTES*documents)
            continue
        if metadata_path is None or not Path(metadata_path).is_file():
            continue
        frame: Any = pd.read_parquet(metadata_path)
        if index == 0:
            # The export's working copy of the table, which the pending publications may still expand by a row per
            # document (the expansion lands in the table and in its copy).
            view_bytes += Path(metadata_path).stat().st_size + 2*METADATA_TABLE_ROW_BYTES*documents
        names: Any = frame['filename'] if 'filename' in frame.columns else frame['filename_base'] if 'filename_base' in frame.columns else None
        if names is None:
            continue
        rows: list[dict[str, Any]] = frame.to_dict(orient='records')
        for name, row in zip(list(names), rows):
            stem = re.sub(r'__p\d{4}-\d{4}$', '', canonical_stem(str(name or '')))
            if stems is None or stem in stems:
                record_bytes += len(json.dumps(row, ensure_ascii=False, default=str).encode())
    return 2*record_bytes + view_bytes


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


# ---- The publication charge (CP-K4c3 R3, calibrated in K4g K4F3-G1) -----------------------------------------------
# Each remaining publication's addition, derived from the writers' own input and product lists; ``remaining_need``
# combines them with the private scratch the run releases between publications (``release_credits``). K4F3-G1: only
# the documents selected for sections are charged to sections/annotation once the selection exists, and the export's
# view and published copy are not counted as present together.
# A conservative charge derived from the writers' own input and product lists. Every publication adds its frozen input
# snapshots, (export) a transient view of the same inputs, its products twice (generation and published copy) and its
# generation manifest; a metadata-writing publication re-publishes the compatibility table three times (input snapshot,
# generation copy, published file). Fixed footprints measured on the Mac (pyarrow 26), rounded up:
PUBLICATION_ORDER = ('download', 'extract', 'clean', 'ocr', 'sections', 'annotate', 'export')
METADATA_STAGES = frozenset(('download', 'extract', 'clean', 'ocr'))  # their wrappers re-publish download_results/*.parquet
METADATA_TABLE_GROWTH_BYTES = 8192  # columns a publication adds beyond the fixed schema footprint (clean: +3.8 KB measured)
CLEANING_REPORT_BYTES = 16384       # cleaning_report.parquet (4.4 KB measured for 1-8 documents) ...
CLEANING_REPORT_ROW_BYTES = 512     # ... plus a row per document
PARQUET_FILE_BYTES = 16384          # one parquet file's schema, footer and statistics (3.9-9.6 KB measured)
PARQUET_ROW_GROUP_BYTES = 4096      # per row group: the sections parquet writes one per document (2.7 KB measured)
SECTION_ROW_BYTES = 1024            # per section row: its JSON fragment keys and computed columns (362 B measured)
ANNOTATION_ROW_BYTES = 1024         # per section row in each annotation parquet and in the lineage manifest
ANNOTATION_PARTITIONS = 20          # GlossSectionClassifier: 5 partitions per CPU, n_cpus=4 (AnnotatePhaseMixin.annotate)
GENERATION_FIXED_BYTES = 8192       # generation.json (0.6-5.6 KB measured) ...
GENERATION_DOCUMENT_BYTES = 1024    # ... plus its per-document (or per-record) entries
SIDECAR_REWRITE_BYTES = 4096        # an extract sidecar/chunk manifest rewritten with compatibility paths
SKIPPED_KINDS = frozenset(('source_snapshot', 'attempt_diagnostic', 'worker_receipt', 'generation_manifest'))
_text_facts: dict[str, tuple[int, int]] = {}  # sha256 -> (JSON-escaped bytes, markdown header lines); content-only facts


def _text_fact(path: Path, sha256: str | None) -> tuple[int, int]:
    import json
    cached = _text_facts.get(sha256) if sha256 else None
    if cached is not None:
        return cached
    try:
        text = Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        # Unreadable now: the worst case of JSON escaping (every byte a \\uXXXX escape) and a section per line.
        size = _size(path)
        return (6*size + 2, size)
    fact = (len(json.dumps(text, ensure_ascii=False).encode()), sum(1 for line in text.splitlines() if line.lstrip().startswith('#')))
    if sha256:
        _text_facts[sha256] = fact
    return fact


def _size(path: Any) -> int:
    try:
        return Path(path).stat().st_size if path is not None and Path(path).is_file() else 0
    except OSError:
        return 0


def _model_bytes(corpus: Any) -> int:
    """The classifier model files the annotation freezes (the manifest and every file beside it, or the model file)."""
    path = Path(getattr(corpus, 'section_classifier_model_path', '') or '')
    if path.is_dir():
        return sum(_size(item) for item in path.rglob('*') if item.is_file())
    if path.is_file():
        if path.suffix == '.json':
            return sum(_size(item) for item in path.parent.iterdir() if item.is_file())
        return _size(path)
    return 0


def _copied(row: Any, settings: dict[str, Any]) -> bool:
    """Whether a compatibility publication copies this product rather than hard-linking it (execution.links_released_copies)."""
    from .execution import RETAINED_PRIVATE_KINDS
    return not (settings.get('debug', 'off') == 'off' and row['retention_class'] == 'intermediate' and row['kind'] not in RETAINED_PRIVATE_KINDS)


def publication_charges(corpus: Any, result: Any, stages: Iterable[str]) -> dict[str, int]:
    """Bytes each remaining publication adds to the corpus filesystem, from the state on disk now (CP-K4c3 R3).

    ``stages``: the publications still to run, in publication order. Inputs and products come from the lists the
    writers use: this run's registered products (compatibility copies), the compatibility tables, the export's frozen
    input list (``export.export_inputs``), the selected texts for sections and the sections parquet and model files
    for the annotation. Nothing is assumed to be released between publications."""
    import re
    from .._naming import canonical_stem
    from .scheduler import local_path
    owner: Any = corpus
    settings: dict[str, Any] = getattr(owner, '_retention_settings', {}) or {}
    remaining = [stage for stage in PUBLICATION_ORDER if stage in set(stages)]
    attempts: dict[str, Any] = result.get('attempts', {})
    by_stage: dict[str, set[Any]] = {}
    for key, attempt in attempts.items():
        by_stage.setdefault(str(key).rsplit(':', 1)[-1], set()).add(attempt)
    produced = set(attempts.values())
    catalog: dict[str, Any] = owner._scheduler_writer.file_catalog()
    present = [row for row in catalog.values() if row['presence'] == 'present' and row['producer_attempt_id'] in produced]
    # One text per document: the clean publication's compatibility copy is a second present row of the same text.
    texts: dict[tuple[Any, Any], Any] = {}
    for row in present:
        if row['kind'] == 'clean_markdown':
            texts.setdefault((row['producer_attempt_id'], row['sha256']), row)
    text_rows = list(texts.values())
    documents = len(text_rows)
    facts = [(row['bytes'] or 0, *_text_fact(local_path(row['uri']), row['sha256'])) for row in text_rows]
    text_bytes = sum(size for size, _, _ in facts)
    # K4F3-G1: only the documents the clean publication accepted for sections are sectioned and annotated. The
    # selection exists once that publication ran in this process (``good_files``); before it, every text counts.
    selected = getattr(owner, 'good_files', None) if 'clean' not in remaining else None
    if isinstance(selected, (list, tuple, set)):
        stems = set(selected)
        section_facts = [fact for fact, row in zip(facts, text_rows) if canonical_stem(local_path(row['uri'])) in stems]
    else:
        section_facts = facts
    section_rows = sum(headers + 1 for _, _, headers in section_facts)
    from .export import metadata_table as selected_table
    try:
        table: Any = selected_table(owner)
    except FileNotFoundError:
        table = None
    table_bytes = _size(table) or None
    projection_bytes = _size((getattr(owner, '_streaming_download_manifest', None) or {}).get('path'))
    charges: dict[str, int] = {}
    for stage in remaining:
        charge = 0
        # Compatibility copies of this run's products that are copied, not linked (debug on, retained kinds).
        for row in present:
            if row['producer_attempt_id'] in by_stage.get(stage, ()) and row['kind'] not in SKIPPED_KINDS and '/.glossapi/' in str(row['uri']):
                if row['kind'] in ('chunk_manifest', 'extract_sidecar'):
                    charge += (row['bytes'] or 0) + SIDECAR_REWRITE_BYTES  # rewritten with compatibility paths
                elif _copied(row, settings):
                    charge += row['bytes'] or 0
        if stage in METADATA_STAGES:
            # The table the stage rewrites, re-published three times. It reaches at most the full compatibility schema's
            # footprint (a download table has fewer columns than the one extraction completes), plus new columns and a
            # row per document.
            # The caller's manifest columns (the source projection) are merged into it.
            table_bytes = max(table_bytes or 0, METADATA_TABLE_FIXED_BYTES) + METADATA_TABLE_GROWTH_BYTES + METADATA_TABLE_ROW_BYTES*documents \
                + projection_bytes
            charge += 3*table_bytes + GENERATION_FIXED_BYTES
            if stage == 'download':
                # The run's own download table (download_results_<manifest>.parquet) besides the canonical one.
                charge += 3*(METADATA_TABLE_FIXED_BYTES + METADATA_TABLE_ROW_BYTES*documents) + GENERATION_FIXED_BYTES
            if stage == 'clean':
                report = _size(Path(owner.output_dir)/'cleaning_report.parquet')
                charge += 3*(report + CLEANING_REPORT_BYTES + CLEANING_REPORT_ROW_BYTES*documents) + GENERATION_FIXED_BYTES
        if stage == 'sections':
            charge += sections_charge(section_facts)
        if stage == 'annotate':
            sections_parquet = Path(getattr(owner, 'sections_parquet', '') or '')
            produced_now = 'sections' not in remaining and sections_parquet.is_file()
            parquet = _size(sections_parquet) if produced_now else sections_parquet_bound(section_facts)
            charge += annotate_charge(parquet, section_rows, _model_bytes(owner))
        if stage == 'export':
            texts_final = not any(earlier in remaining for earlier in ('download', 'extract', 'clean', 'ocr'))
            charge += export_charge(owner, text_rows, facts, table_bytes, table, produced, texts_final=texts_final)
        charges[stage] = int(charge)
    return charges


def sections_parquet_bound(facts: list[tuple[int, int, int]]) -> int:
    """Upper bound of ``sections_for_annotation.parquet``: the section text uncompressed (parquet stores strings as plain
    bytes plus a length; it never escapes), per-row columns, a row group per document and the file's schema footprint."""
    return PARQUET_FILE_BYTES + sum(size + PARQUET_ROW_GROUP_BYTES + SECTION_ROW_BYTES*(headers + 1) for size, _, headers in facts)


def sections_charge(facts: list[tuple[int, int, int]]) -> int:
    """The sections publication: per document a text snapshot, its JSON fragment (escaped text plus per-row keys) and a
    generation manifest; the aggregate's snapshot of every fragment; the parquet twice; its generation manifest."""
    fragments = sum(escaped + SECTION_ROW_BYTES*(headers + 1) for _, escaped, headers in facts)
    per_document = sum(size for size, _, _ in facts) + fragments + GENERATION_DOCUMENT_BYTES*len(facts)
    return per_document + fragments + 2*sections_parquet_bound(facts) + GENERATION_FIXED_BYTES + GENERATION_DOCUMENT_BYTES*len(facts)


def annotate_charge(sections_parquet: int, rows: int, model: int) -> int:
    """The annotation batch: snapshots of the sections parquet and the model files; the classified and fully annotated
    parquets and the classifier's fixed partition files, each twice; a lineage manifest with an entry per row."""
    data = sections_parquet + ANNOTATION_ROW_BYTES*rows
    products = 2*(data + PARQUET_FILE_BYTES) + ANNOTATION_PARTITIONS*PARQUET_FILE_BYTES + data
    return sections_parquet + model + 2*products + GENERATION_FIXED_BYTES + ANNOTATION_ROW_BYTES*rows


def export_charge(corpus: Any, text_rows: list[Any], facts: list[tuple[int, int, int]], table_bytes: int | None,
                  table: Any, produced: set[Any], *, texts_final: bool) -> int:
    """The export publication: its frozen input snapshots and the view of the same inputs (``export.export_inputs``,
    the writer's own list), the records twice (generation and published copy) and the generation manifest.

    Records: escaped text plus ``RECORD_OVERHEAD_BYTES`` per document, plus the planned documents' metadata rows (and
    the run's source projection, an upper bound of the merged rows). Before the texts are published, the inputs are
    this run's texts, the table as the metadata publications will leave it, and the sidecars the serializer reads."""
    import importlib
    import json
    import re
    from .._naming import canonical_stem
    from .scheduler import local_path
    owner: Any = corpus
    planned: dict[Any, str] | None = getattr(owner, '_streaming_publication_stems', None)
    stems = set(planned.values()) if planned else None
    inputs_bytes = 0
    metadata_path: Any = None
    inputs: list[Any] = []
    selected: list[Any] = []
    if texts_final:
        try:
            from .export import export_inputs
            inputs, _, selected = export_inputs(owner, {})
        except FileNotFoundError:
            inputs, selected = [], []
    if inputs:
        # Every text and table publication is done: exactly the writer's list.
        inputs_bytes = sum(_size(path) for _, path in inputs)
        metadata_path = inputs[0][1]
        record_text = sum(_text_fact(path, None)[0] for _, path in selected) + RECORD_OVERHEAD_BYTES*len(selected)
        documents = len(selected)
    else:
        # Before the publications: this run's texts and sidecars, the table the metadata publications will write.
        record_text = sum(escaped for _, escaped, _ in facts) + RECORD_OVERHEAD_BYTES*len(facts)
        documents = len(facts)
        inputs_bytes = sum(size for size, _, _ in facts) + (table_bytes if table_bytes is not None else METADATA_TABLE_FIXED_BYTES + METADATA_TABLE_ROW_BYTES*documents)
        names = {canonical_stem(local_path(row['uri'])) for row in text_rows}
        catalog: dict[str, Any] = owner._scheduler_writer.file_catalog()
        for row in catalog.values():
            name = Path(str(row['uri'])).name
            if row['presence'] == 'present' and row['producer_attempt_id'] in produced and name.endswith(('.metrics.json', '.latex_map.jsonl')):
                inputs_bytes += row['bytes'] or 0
        for stem in names:
            for path in (Path(owner.output_dir)/'json/metrics'/f'{stem}.metrics.json', Path(owner.output_dir)/'json/metrics'/f'{stem}.per_page.metrics.json',
                         Path(owner.output_dir)/'json'/f'{stem}.latex_map.jsonl'):
                inputs_bytes += _size(path)
        metadata_path = table
    row_bytes = 0
    pd: Any = importlib.import_module('pandas')
    projection: dict[str, Any] = getattr(owner, '_streaming_download_manifest', None) or {}
    for path in (metadata_path, projection.get('path')):
        if path is None or not Path(path).is_file():
            continue
        frame: Any = pd.read_parquet(path)
        names: Any = frame['filename'] if 'filename' in frame.columns else frame['filename_base'] if 'filename_base' in frame.columns else None
        if names is None:
            continue
        for name, row in zip(list(names), frame.to_dict(orient='records')):
            stem = re.sub(r'__p\d{4}-\d{4}$', '', canonical_stem(str(name or '')))
            if stems is None or stem in stems:
                row_bytes += len(json.dumps(row, ensure_ascii=False, default=str).encode())
    if metadata_path is None or not Path(metadata_path).is_file():
        row_bytes += METADATA_TABLE_ROW_BYTES*documents
    records = record_text + row_bytes
    # The frozen input snapshots and the generation stay; the transient view of the inputs is removed before the
    # publication copies the generation to export.jsonl, so the peak holds the larger of the two, not both (K4F3-G1).
    return inputs_bytes + records + max(inputs_bytes, records) + GENERATION_FIXED_BYTES + GENERATION_DOCUMENT_BYTES*documents


RELEASED_AFTER = {'clean': 'extract', 'sections': 'clean', 'annotate': 'sections'}  # run.py: retained after each publication


def release_credits(corpus: Any, result: Any, *, honour_suspension: bool = True) -> dict[str, int]:
    """Bytes released right after each publication (K4F3-G1): the run retires the private scratch of the streaming
    attempts the publication was the last reader of (``run.py`` calls ``retain_completed_run`` for
    ``RELEASED_AFTER[stage]``). Counted from the same candidate rule retention applies (``_retention_candidates``, minus
    the private rows ``_collect_candidates`` keeps), only for files present now whose every hard link is released and
    whose planned readers are done by then; nothing is credited with ``debug=on``, which keeps them."""
    from .retention import _retention_candidates, document_releases_suspended, file_path, resolved_settings
    owner: Any = corpus
    settings = resolved_settings({**(getattr(owner, '_retention_settings', {}) or {})})
    if settings.get('debug') == 'on':
        return {}
    writer: Any = owner._scheduler_writer
    # The same rule run.py applies before each release (CP-K4g F1): while a document is failed, none happens, so
    # none is credited. ``honour_suspension=False`` only measures what an unsuspended run would release.
    if honour_suspension and document_releases_suspended(writer, result['run_id']):
        return {}
    root = Path(owner.output_dir)
    private = root.resolve()/'.glossapi/artifacts'
    attempts: dict[str, Any] = result.get('attempts', {})
    credits: dict[str, int] = {}
    tasks: dict[str, Any] = writer.control()['tasks']
    for publication, stage in RELEASED_AFTER.items():
        chosen = [attempt for key, attempt in attempts.items() if str(key).endswith(':'+stage)]
        if not chosen:
            continue
        done_by = PUBLICATION_ORDER.index(publication)
        # Bytes are freed only when every link of a file goes: a released private product is often hard-linked to its
        # public compatibility copy (``link_published``), so an inode counts only when all its links are released here.
        inodes: dict[tuple[int, int], tuple[int, int, set[str]]] = {}
        for row in _retention_candidates(writer, root, result['run_id'], settings, attempts=chosen):
            path = file_path(row)
            if path.is_relative_to(private) and (row['retention_class'] != 'intermediate' or row['kind'] == 'download_payload'
                    or (settings['memory_saving'] == 'extreme' and row['kind'] == 'source_snapshot')):
                continue
            # release_private_scratch (and delete_many) release a file only after every planned reader finished: a
            # reader that is a later publication (the export reading the cleaned texts) keeps it past this release.
            if not _readers_done_by(writer, result['run_id'], row, tasks, done_by):
                continue
            try:
                info = os.lstat(path)
            except OSError:
                continue
            key = (info.st_dev, info.st_ino)
            entry = inodes.setdefault(key, (info.st_size, info.st_nlink, set()))
            entry[2].add(str(path))
        total = sum(size for size, links, paths in inodes.values() if len(paths) >= links)
        credits[publication] = total
    return credits


def _readers_done_by(writer: Any, run_id: str, row: Any, tasks: dict[str, Any], done_by: int) -> bool:
    """Every planned reader of ``row`` has finished, or belongs to a publication at or before ``PUBLICATION_ORDER[done_by]``."""
    for reader in writer.planned_readers(run_id, row):
        status, reason = writer.reader_state(reader)
        if status == 'ok' or (status == 'skipped' and reason == 'not_in_plan'):
            continue
        stage = (tasks.get(reader) or {}).get('stage')
        if stage not in PUBLICATION_ORDER or PUBLICATION_ORDER.index(stage) > done_by:
            return False
    return True


def remaining_need(charges: dict[str, int], credits: dict[str, int]) -> int:
    """The corpus-filesystem bytes the remaining publications need at their peak: before each publication t, every
    earlier remaining publication's addition is present and every release that followed one of them has happened."""
    need = present = 0
    for stage in PUBLICATION_ORDER:
        if stage not in charges:
            continue
        need = max(need, present + charges[stage])
        present += charges[stage] - credits.get(stage, 0)
    return max(need, 0)


def check_exact(corpus: Any, result: Any, stages: Iterable[str], *, when: str) -> None:
    """The binding check (K4BI-G7, CP-K4a F7, CP-K4ac2 R3, CP-K4c3 R3), extreme only: the remaining publications'
    charge (``publication_charges``) against the corpus filesystem's room now.

    Called before the first publication with every requested stage, and again before each publication with that stage
    and every later one."""
    from .disk_budget import DiskBudget
    owner: Any = corpus
    settings: dict[str, Any] = getattr(owner, '_retention_settings', {}) or {}
    if settings.get('memory_saving') != 'extreme' or not result.get('run_id'):
        return
    remaining = [stage for stage in PUBLICATION_ORDER if stage in set(stages)]
    charges = publication_charges(owner, result, remaining)
    require_room(owner.output_dir, settings, remaining_need(charges, release_credits(owner, result)), when=when,
                 present=DiskBudget(owner._scheduler_writer, owner.output_dir, result['run_id'], settings).present_bytes())
