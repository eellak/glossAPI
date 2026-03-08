# Clean Stage

## Purpose

The clean stage normalizes extracted Markdown and evaluates its quality.

## Main responsibilities

- run Rust-backed cleaning
- compute text quality and badness metrics
- detect documents that require OCR reruns
- update metadata for downstream stage selection

## Main inputs

- Markdown in `markdown/`
- metadata parquet
- cleaner configuration and thresholds

## Main outputs

- cleaned Markdown in `clean_markdown/`
- quality metrics and reports
- metadata updates including OCR-related decisions

## Why this stage is critical

This stage is the main bridge between extraction and corrective OCR.

It is especially important for Greek corpora because it distinguishes:

- technically extracted text
- actually usable Greek text

## Important operational outputs

This stage may contribute or update:

- quality scores
- filter classifications
- `needs_ocr`
- character-count-based diagnostics
- processing-stage status

## Failure concerns

Typical issues include:

- missing Rust extension
- metadata bootstrap issues
- misleadingly non-empty but low-quality Markdown

## Contributor note

Changes here affect OCR routing and post-run quality analysis. Treat score and flag semantics as contract-level behavior.
