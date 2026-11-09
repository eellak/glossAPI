# Corpus-wide barriers of a streaming run

A streaming run (`glossapi run`, `Corpus.run`) processes each document through its stages as soon as its inputs exist.
These are the only points where it waits for the whole corpus. `plan --json` previews exactly these rows for the planned
stages (`barriers`, with `executed_as`, `why` and `continues`), and the run executes them as
`<stage>.compatibility_publish` tasks, each chained after the previous one (`corpus_seal` dependencies), plus the
source-manifest seal when the plan is frozen. `tests/test_ledger_w18_barriers.py` keeps the code, this table and the
preview equal.

<!-- generated from glossapi.stages.barriers.markdown_table(); do not edit by hand -->
| barrier | brought by stage | executed as | why it waits | what continues |
| --- | --- | --- | --- | --- |
| `source_manifest` | always | `freeze` | The full requested source set and its expansion order fix download names and URL duplicates before any worker starts. | Nothing runs before the seal; afterwards download, extract, clean and sections stream per document. |
| `download_publication` | download | `download.compatibility_publish` | download_results/ and downloads/ are written by the unchanged single-writer download serializer over the sealed manifest; it replays committed acquisitions and never fetches. | Every document task (download, extract, clean, OCR, sections) runs before it; under memory_saving=extreme each owned download is already deleted after its last document reader. |
| `extract_publication` | extract | `extract.compatibility_publish` | markdown/, sidecars/extract/ and repaired_pdfs/ are published by the legacy extract wrapper, which replays committed extractions in one pass and merges per-document repair metadata once. | Document extraction, cleaning, OCR and section parsing all finish before it; a download deleted under memory_saving=extreme is bound by its recorded hash and not read again. |
| `clean_publication` | clean | `clean.compatibility_publish` | clean_markdown/ and the cleaning report are written by the directory-level cleaner wrapper over the published raw markdown. | Per-document cleaning already ran in the DAG; this publishes its committed outputs. |
| `ocr_publication` | ocr | `ocr.compatibility_publish` | OCR markdown is materialized and the cleaner refreshed over markdown/ by one publisher (OCR success metadata is a corpus parquet). | Per-document OCR and its re-clean already ran in the DAG. |
| `sections_selection_ids` | sections | `sections.compatibility_publish` | Legacy section selection and global section IDs use corpus traversal order and the aggregate parquet needs the selected set. | Per-document section parsing streams in the DAG; only selection, IDs and the aggregate parquet wait. |
| `annotation_batch` | annotate | `annotate.compatibility_publish` | Annotation predicts over PyArrow partitions, then materializes whole-corpus frames for post-processing and full annotation. | Every document's sections are parsed before it; annotation itself does not stream yet. |
| `export_shards` | export | `export.compatibility_publish` | Ordered export membership and shard boundaries need the sealed selected set; every selected version gets an inclusion or omission receipt. | Record preparation inputs are complete; the single publisher seals one generation. |
<!-- end generated -->

Not barriers:
- Durability: before any unlink a user or a later stage could need, the run makes the needed bytes and ledger rows durable
  with targeted fsyncs of the affected files and directories and a FULL ledger commit (K4(c) part 1; never `syncfs`).
  These are short waits inside a step, not corpus-wide waiting points.
- Optional content-duplicate finalization is record-only today: it changes no membership and executes no barrier.
- Directory-level cleaning, repair, routing, OCR and math kernels run per document inside the DAG; their directory
  wrappers appear above only as publications of committed results.
