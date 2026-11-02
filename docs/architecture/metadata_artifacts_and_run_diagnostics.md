# Metadata, Artifacts, and Run Diagnostics

GlossAPI preserves both source metadata and pipeline-generated metadata.

This is a central feature of the system, not a side effect.

## Two kinds of metadata

The pipeline needs to retain:

1. original metadata from scraping or source discovery
2. pipeline metadata generated during processing

Original metadata helps preserve context such as:

- source URL
- filename assignment context
- source collection information
- document type or source-specific attributes

Pipeline metadata helps answer:

- what happened to this document during processing
- which stages ran
- what quality issues were detected
- whether the document was skipped, failed, repaired, or exported

## Why both matter

Without source metadata, the corpus loses provenance.

Without pipeline metadata, the run becomes difficult to debug or analyze.

GlossAPI needs both because it is not only extracting text. It is curating a processing history.

## Metadata as the state backbone

The metadata parquet is a key state object in the current architecture.

It is used to:

- preserve prior results across reruns
- select OCR candidates
- track processing stage progression
- join source context with derived quality information
- support post-run analytics

This is why metadata updates should be treated carefully and documented whenever field semantics change.

In a streaming run (`Corpus.run()`, `glossapi run`) the authority is the ledger `corpus.sqlite3`; the metadata
parquet (`download_results/download_results.parquet`) is a compatibility table the run publishes for the phase
methods and for callers that read it. The rules below apply to that table.

### Rows without a file name

A skipped duplicate URL and a failed download have an empty `filename`. Metadata merges never key such rows by file
name: each one is kept as its own row, with its own URL, error and source fields. When the download results are
folded into an existing `download_results.parquet`, a row without a file name replaces the canonical row without a
file name that records the same URL, or is added. A file name that appears on several rows receives the artifact
evidence (extraction, metrics) on each of them.

### Column types

Metadata unions keep every value exactly. Integer columns stay integers when rows are added (values above 2**53 are
never rounded); a column becomes the nullable `Int64` (or `UInt64`) type only when a float could not hold its values.
A union that would turn a categorical, a time-zone-aware datetime, a period or an interval column into plain objects
is refused with `MetadataSchemaError` before anything is written; two categoricals with different categories are
merged into one categorical. Decimal and list values are stored as the parquet serializer stores them (objects);
their exact Python type is not guaranteed on read.

### Which metadata parquet each phase reads

Two read rules exist. `Corpus.clean` looks first for a metadata parquet in `input_dir`, then ensures
`output_dir/download_results/download_results.parquet`, then one in `input_dir`. `Corpus.section`, `Corpus.ocr` and
math enrichment ensure the output's canonical table first, then search `output_dir`, then `input_dir` and
`input_dir/download_results`. A `Corpus` object remembers the first parquet it resolved (or the `metadata_path` given
to its constructor), and every later phase of that object uses it. When `input_dir` contains its own metadata
parquet, call the phases on one `Corpus` object, or pass `metadata_path`, so that every phase uses the same file.

### Math enrichment flags

`enriched_math` / `math_enriched` in the metadata parquet is a compatibility projection: it is true once math
enrichment ran for the document, including a run that found no formula (`math_items = 0`). It does not distinguish an
empty result from a failure; `glossapi status` and `glossapi report` (the ledger) do.

## Diagnostic information that should survive the run

A good corpus run should leave enough metadata and artifacts to analyze:

- downloads that succeeded or failed
- extraction outputs that were produced
- files marked as low quality
- files rerouted through OCR
- files skipped because of known issues
- files that reached sectioning or annotation

This supports both debugging and corpus-quality assessment.

## Blockers, issues, and post-run analysis

One of the design goals of the pipeline is to recognize blockers and stage issues for later analysis.

That means the system should make it possible to distinguish:

- success
- recoverable failure
- skip
- timeout
- quality failure
- rerun candidate
- downstream blocker

If these states become ambiguous, post-run analysis quality drops sharply.

## Artifacts as evidence

Retained artifacts matter because they are often the easiest way to inspect what happened:

- raw downloaded files
- extracted Markdown
- cleaned Markdown
- Docling JSON
- metrics JSON
- section parquet outputs
- logs

These artifacts complement metadata rather than replacing it.

## Documentation guidance

Each stage should document:

- which metadata it reads
- which metadata it writes
- which artifacts it creates
- which statuses or issue classes it can introduce

This makes it possible to reason about the run as a whole instead of as isolated methods.

## Long-term design note

The current design mixes:

- metadata as durable logical state
- artifacts as operational evidence
- some filesystem-based inference

That works well today, but it is also a useful place to identify future refactoring opportunities:

- more explicit document identity
- clearer status schemas
- better retention classes for artifacts
- less inference from directory scanning alone
