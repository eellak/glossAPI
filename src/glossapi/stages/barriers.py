"""Every corpus-wide waiting point a streaming run executes (W18 M13, R34).

One table is the source of truth: ``plan --json`` previews these rows for the planned stages, the run executes
them (``<stage>.compatibility_publish`` tasks chained by ``corpus_seal`` dependencies, plus the source-manifest
seal at freeze), and docs/architecture/corpus_barriers.md lists them. A test keeps the three equal.
"""
from __future__ import annotations

# name, stage that brings it (None: always), what the run executes, why it waits, what continues meanwhile.
BARRIERS: tuple[dict[str, str | None], ...] = (
    dict(name='source_manifest', stage=None, executed_as='freeze',
         why='The full requested source set and its expansion order fix download names and URL duplicates before any worker starts.',
         continues='Nothing runs before the seal; afterwards download, extract, clean and sections stream per document.'),
    dict(name='download_publication', stage='download', executed_as='download.compatibility_publish',
         why='download_results/ and downloads/ are written by the unchanged single-writer download serializer over the sealed manifest; '
             'it replays committed acquisitions and never fetches.',
         continues='Every document task (download, extract, clean, OCR, sections) runs before it; under memory_saving=extreme each owned '
                   'download is already deleted after its last document reader.'),
    dict(name='extract_publication', stage='extract', executed_as='extract.compatibility_publish',
         why='markdown/, sidecars/extract/ and repaired_pdfs/ are published by the legacy extract wrapper, which replays committed '
             'extractions in one pass and merges per-document repair metadata once.',
         continues='Document extraction, cleaning, OCR and section parsing all finish before it; a download deleted under '
                   'memory_saving=extreme is bound by its recorded hash and not read again.'),
    dict(name='clean_publication', stage='clean', executed_as='clean.compatibility_publish',
         why='clean_markdown/ and the cleaning report are written by the directory-level cleaner wrapper over the published raw markdown.',
         continues='Per-document cleaning already ran in the DAG; this publishes its committed outputs.'),
    dict(name='ocr_publication', stage='ocr', executed_as='ocr.compatibility_publish',
         why='OCR markdown is materialized and the cleaner refreshed over markdown/ by one publisher (OCR success metadata is a corpus '
             'parquet).',
         continues='Per-document OCR and its re-clean already ran in the DAG.'),
    dict(name='sections_selection_ids', stage='sections', executed_as='sections.compatibility_publish',
         why='Legacy section selection and global section IDs use corpus traversal order and the aggregate parquet needs the selected set.',
         continues='Per-document section parsing streams in the DAG; only selection, IDs and the aggregate parquet wait.'),
    dict(name='annotation_batch', stage='annotate', executed_as='annotate.compatibility_publish',
         why='Annotation predicts over PyArrow partitions, then materializes whole-corpus frames for post-processing and full '
             'annotation.',
         continues='Every document\'s sections are parsed before it; annotation itself does not stream yet.'),
    dict(name='export_shards', stage='export', executed_as='export.compatibility_publish',
         why='Ordered export membership and shard boundaries need the sealed selected set; every selected version gets an inclusion or '
             'omission receipt.',
         continues='Record preparation inputs are complete; the single publisher seals one generation.'),
)


def planned_barriers(stages, remaining):
    """The barrier rows a run with these planned stages executes, in execution order (the ``plan --json`` preview)."""
    stages = set(stages)
    return [dict(name=str(row['name']), scope='corpus', waiting_for='sealed_selection' if row['stage'] is None else 'every_document_task_terminal',
                 remaining=remaining, executed_as=str(row['executed_as']), why=str(row['why']), continues=str(row['continues']))
            for row in BARRIERS if row['stage'] is None or row['stage'] in stages]


def executed_as(name):
    return next(row['executed_as'] for row in BARRIERS if row['name'] == name)


def markdown_table():
    """The documentation table (docs/architecture/corpus_barriers.md)."""
    lines = ['| barrier | brought by stage | executed as | why it waits | what continues |', '| --- | --- | --- | --- | --- |']
    lines += [f"| `{row['name']}` | {row['stage'] or 'always'} | `{row['executed_as']}` | {row['why']} | {row['continues']} |" for row in BARRIERS]
    return '\n'.join(lines)
