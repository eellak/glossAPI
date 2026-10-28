# Extract Stage

## Purpose

The extraction stage converts source files into Markdown and optional structured byproducts such as Docling JSON.

## Main responsibilities

- locate source files for processing
- choose extraction backend behavior
- produce Markdown
- optionally emit structured JSON artifacts
- balance throughput against stability

## Main inputs

- files in `downloads/` or explicitly provided file paths
- extraction configuration
- backend selection and accelerator settings

## Main outputs

- repaired PDF sidecars in `repaired_pdfs/<stem>.pdf` when eligible
- font-repair metadata in the canonical download-results parquet
- Markdown in `markdown/`
- optional JSON artifacts in `json/`
- stage logs and timeout/problematic-file traces

## Key design point

This stage is where the Docling throughput strategy meets the pipeline's stability constraints.

It should be understood as both:

- a text extraction stage
- an operational risk management stage

## Backend semantics

The extraction logic distinguishes between safer extraction behavior and Docling-focused behavior.

This matters because Docling is powerful, but not equally stable under all batching and corpus conditions.

## Font repair pre-pass

`Corpus.extract(..., repair_fonts=True)` enables CPU ToUnicode repair by default
for both the safe and Docling backends. Set `repair_fonts=False` to extract the
original text layer. Each file's extraction worker analyzes its PDF immediately
before conversion; a `repairable` or `mixed` document with at least one eligible
font receives a repaired copy in `<output_dir>/repaired_pdfs/<stem>.pdf`. The
extractor opens that copy. Original PDFs in `downloads/` stay unchanged, and
Markdown names, resume keys and metadata retain the original filename/stem.
Non-PDF inputs bypass repair.

A repaired copy newer than its source is reused without analysis or rewriting.
Ordinary `skip_existing=True` resume still skips completed Markdown entirely;
use `skip_existing=False` to re-extract while reusing a fresh repaired PDF.

The pre-pass uses at most half of the existing per-file extraction timeout,
reserving the rest for extraction. Its elapsed time is deducted from the shared
budget, including chunked conversion and retries. Analysis/repair failures and
pre-pass timeouts fall back to the original PDF and record `error` plus the
message in the extract sidecar and parquet's existing `extract_error` column.
The stage continues. Docling retains its existing native timeout behavior;
converters without native timeout support cannot enforce the extraction portion
of the deadline. POSIX main-thread signal timers bound the pre-pass without an
extra process pool. Unsupported timer contexts fall back with `error`; native C
calls may delay signal delivery until they return to Python.

Workers write `sidecars/extract/<stem>.json`; the controller merges these four
columns into the canonical metadata parquet after extraction:

| Column | Type / default | Meaning |
| --- | --- | --- |
| `font_repair_status` | string / `skipped` | `repaired`, `not_needed` (consistent), `unsupported`, `no_text_layer`, `error`, or `skipped` (disabled/unprocessed). |
| `font_repair_fonts` | int / `0` | Number of fonts repaired; reused copies retain the saved count when available. |
| `font_repair_valid_rate_before` | float / null | Dictionary validity over the original PDF's first 20 pages. |
| `font_repair_valid_rate_after` | float / null | Dictionary validity over the repaired PDF's first 20 pages, or the unchanged original. |

Configure `GLOSSAPI_GREEK_WORDLIST` with a local UTF-8 word list to compute rates
with `extract_text_stats`. Missing/unreadable dictionaries or metric failures
leave rates null. A reused copy without saved repair metadata has an unknown
font count represented by `0`; reuse still skips font analysis. Each repaired
PDF gets one INFO line with its stem, font count and available before→after
rates. The final summary counts statuses for the stage's selected PDFs, including
previously completed files on resume.

## Failure concerns

Typical issues include:

- Docling crashes
- hangs or timeouts
- worker poisoning after a bad file
- partial progress during corpus-scale processing

## Recovery concerns

Recovery should preserve:

- successful Markdown already produced
- visibility into failed documents
- enough information to avoid repeating the same crash pattern blindly

## Contributor note

Any change to batch policy, backend selection, or output naming should be treated as an architectural change, not just a local refactor.
