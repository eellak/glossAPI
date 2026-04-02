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

## DeepSeek runtime contract

- `ocr()` may execute page-range shards internally when `use_gpus="multi"` and `scheduler="exact_fill"`, but the stage contract remains one canonical Markdown file and one canonical metrics file per source PDF.
- When shard execution is used, the runner reassembles `markdown/<stem>.md` and `json/metrics/<stem>.metrics.json` after the CLI workers finish.
- Execution-time shard artifacts are moved under `sidecars/ocr_shards/` so downstream stages do not mistake them for canonical stage outputs.
- The vLLM runtime now streams rendered pages through an in-memory queue, overlaps rendering with inference, skips empty pages before inference, and reuses the same in-memory image for repair retries.
- Canonical OCR markdown now annotates page boundaries with `<!-- page:N -->` comments alongside each page-split marker so downstream inspection can line up page images and markdown more easily.
- In `repair_mode="auto"`, a page that trips the garbage cutoff again during the plain-OCR repair pass is now blanked instead of keeping the original garbage text.
- Multi-GPU vLLM runs now execute through a durable shared batch queue rather than one fragile subprocess per preassigned lane. Workers claim first-pass batches dynamically, heartbeat while a batch is active, and can be respawned without losing finished batch outputs.
- Repair retries are now durable too. Flagged pages are published back into the same runtime database as a second global repair queue, and any GPU worker can drain those repair shards after the first-pass queue is complete.
- By default each durable batch gets at most two total attempts, so one retry is allowed after the first failure and then the batch is marked failed for operator follow-up.
- Operational sidecars for these runs live under `sidecars/ocr_runtime/`, including the durable work queue state, per-worker runtime JSON, GPU telemetry samples, GPU preflight output, and a final runtime summary with steady-state inference timestamps.

## Contributor note

Any change to candidate selection, skiplist semantics, or OCR-success metadata affects both rerun behavior and corpus analysis quality.
