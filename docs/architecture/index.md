# Architecture Overview

GlossAPI is a staged corpus-processing pipeline centered around the `Corpus` API.

Core extraction uses PDFium with Rust layout_v2 labels. `Corpus.run()` and the CLI execute a ledger-backed DAG with frozen settings and retention; individual phase methods retain compatibility filesystem/parquet behavior. Docling layout/math is optional, and OCR uses the pinned Greek toolkit.

GlossAPI therefore has two interfaces over the same stages:

- **Streaming runs** (`Corpus.run()`, `glossapi plan` / `glossapi run`): one ledger-backed DAG recorded in
  `corpus.sqlite3`, with frozen settings, per-task attempts and retries, and retention of intermediates. See
  [Resumability, Recovery, and Retention](resumability_recovery_and_retention.md) and
  [Corpus-wide barriers](corpus_barriers.md).
- **Phase methods** (`download()`, `extract()`, `clean()`, `ocr()`, `section()`, `annotate()`, `jsonl()`,
  `jsonl_sharded()`): individual calls that keep the compatibility behavior described below, where stages hand off
  through the output directory, filenames and the metadata parquet.

At a high level, the stages are:

1. `download()`
2. `extract()`
3. `clean()`
4. `ocr()`
5. `section()`
6. `annotate()`
7. export: `jsonl()` / `jsonl_sharded()`, or the streaming `export` stage

Not every run uses every stage, but the stages are designed to compose through a shared output directory and shared metadata.

## Why the pipeline is staged

The staging is deliberate. Each stage exists because academic PDFs, especially Greek-language PDFs, fail in different ways:

- downloads may fail, duplicate, or need retry logic
- extraction may succeed structurally but produce unusable text
- many Greek PDFs have broken or missing `ToUnicode` mappings
- OCR is expensive, so it should be applied selectively
- sectioning and annotation should operate only on documents considered usable
- corpus analysis requires retaining what happened at each stage, not only the final text

The pipeline therefore separates:

- acquisition
- text extraction
- text quality validation
- remediation via OCR
- structural segmentation
- annotation
- downstream export

## Main orchestration surface

The main user-facing object is `Corpus` in `src/glossapi/corpus/corpus_orchestrator.py`.

`Corpus` wires together phase mixins:

- `DownloadPhaseMixin`
- `ExtractPhaseMixin`
- `CleanPhaseMixin`
- `OcrMathPhaseMixin`
- `SectionPhaseMixin`
- `AnnotatePhaseMixin`
- `ExportPhaseMixin`

This is the main organizational boundary in the codebase.

## Phase summary

### 1. Download

Purpose:

- fetch files from URL-bearing parquet inputs
- preserve scraping metadata
- produce local files plus download result metadata

Important characteristics:

- async downloader
- retry and rate-limit behavior
- resume support from prior parquet results
- filename normalization and collision avoidance

### 2. Extract

Purpose:

- turn input documents into Markdown
- with the optional Docling backend only, emit Docling JSON and formula index artifacts

Important characteristics:

- the default safe backend uses PDFium and the bundled Rust extensions to write `layout_v2` Markdown, with no Docling
  or GPU (`GLOSSAPI_MARKDOWN_LAYOUT=plain` selects plain page text)
- the Docling backend is optional (the `layout` package); select it explicitly with `phase1_backend='docling'` when
  you need Docling layout JSON for math/code enrichment or Docling layout throughput
- backend selection is part performance decision, part stability decision

### 3. Clean

Purpose:

- clean Markdown
- compute quality and badness metrics
- identify documents that should be rerouted to OCR

Important characteristics:

- uses Rust extension for speed and consistency
- writes metrics back into metadata parquet
- provides operational gating for downstream stages

### 4. OCR

Purpose:

- OCR the physical pages that page quality routes, and optionally run math/code enrichment from existing Docling JSON

Important characteristics:

- OCR uses the pinned Greek OCR toolkit (an optional install); retained native text is kept for pages that are not
  routed
- reads metadata to find OCR candidates: by default (`page_routing='all'`) routed pages in flagged and unflagged
  documents; `page_routing='flagged'` restricts OCR to cleaner-flagged documents
- the phase method is skiplist-aware; a streaming run records task attempts and retries in the ledger instead
- designed as a corrective stage, not the default for every document

### 5. Section

Purpose:

- split Markdown documents into structural sections for classification and downstream analysis

Important characteristics:

- prefers documents that passed quality checks or were repaired successfully
- writes parquet section records

### 6. Annotate

Purpose:

- classify sections with the packaged ML model
- optionally produce fully annotated section outputs

### 7. Export

Purpose:

- produce downstream JSONL- and parquet-friendly outputs
- aggregate metadata from chunked and unchunked runs

## What makes GlossAPI operationally distinct

Four themes dominate the current design:

1. Core extraction needs neither Docling nor a GPU; Docling is an optional backend for layout JSON, math/code
   enrichment and layout throughput.
2. Optional backends (Docling, OCR) can be unstable under batching and parallelism, so recovery logic matters.
3. Greek text validity is a first-class quality concern because raw extraction is often wrong even when it does not crash.
4. Metadata is treated as a durable record of both original scraping context and pipeline-generated diagnostics.

Two more themes are equally important for maintainers:

1. resumability and recovery semantics
2. filesystem artifact and state contracts

## Typical run shape

For a practical corpus run, the intended flow is often:

1. download the corpus
2. extract to Markdown
3. clean and score text quality
4. OCR the pages page quality routes (in flagged and unflagged documents by default)
5. section the resulting usable corpus
6. optionally annotate and export

This allows expensive OCR to be focused on genuinely problematic files.

## Current architectural pressure points

The current architecture is effective but has important tradeoffs:

- stage handoff depends partly on method calls and partly on filesystem conventions
- the phase methods resume from file presence and metadata parquet state; a streaming run (`Corpus.run()`, `glossapi run`) resumes from the ledger `corpus.sqlite3` instead
- phase methods keep their intermediate artifacts, which improves auditing but increases storage; streaming runs apply
  the retention settings recorded in the ledger
- filename-based identity is practical but puts pressure on future chunking and storage redesigns

These pressure points are documented separately in:

- [Artifact Layout and Stage Handoffs](artifact_layout_and_stage_handoffs.md)
- [OCR Cleaning Runtime](ocr_cleaning_runtime.md)
- [Resumability, Recovery, and Retention](resumability_recovery_and_retention.md)
- [Markdown Library Survey](markdown_library_survey.md) — design rationale for the parser-backed Phase A (Pilot B).
