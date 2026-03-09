# OCR Stage

## Purpose

The OCR stage repairs documents whose extracted text is considered unreliable, and can optionally perform math-aware enrichment depending on backend and configuration.

## Main responsibilities

- identify OCR candidates from metadata
- respect skiplist and prior completion state
- rerun problematic files through an OCR backend
- update stage metadata and corrected outputs

## Main inputs

- metadata parquet with OCR-related flags
- source files and/or extracted artifacts
- backend-specific environment and model configuration

## Main outputs

- corrected Markdown or OCR-enriched outputs
- backend-specific JSON or related artifacts
- metadata updates such as OCR success markers

## Backend choices

The supported OCR remediation backend is DeepSeek OCR. Docling remains part of
the surrounding extraction and layout flow, but OCR reruns themselves are now
expected to use the DeepSeek runtime.

## Selection model

OCR is generally selective. The stage is expected to run on documents flagged by earlier quality checks rather than on the entire corpus.

## Recovery concerns

OCR reruns should preserve:

- prior quality diagnostics
- explicit indication that remediation was attempted
- visibility into files that remain problematic

## Contributor note

Any change to candidate selection, skiplist semantics, or OCR-success metadata affects both rerun behavior and corpus analysis quality.
