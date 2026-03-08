# Resumability, Recovery, and Retention

GlossAPI is designed to process corpora that are too large and too messy to assume a perfect one-pass run.

This document covers three related ideas:

1. resumability
2. recovery from failures
3. retention of artifacts and state over time

## Resumability

Resumability means the pipeline should be able to restart without discarding already successful work.

In the current architecture this depends on:

- retained stage artifacts on disk
- metadata parquet updates
- previously recorded success and failure states
- naming conventions that let the pipeline reconnect outputs

Resumability is especially important for:

- large corpora
- unstable extraction/OCR environments
- long-running download jobs
- iterative reruns after bug fixes or environment fixes

## Recovery

Recovery means the pipeline should degrade safely when a subset of documents fail.

Typical recovery situations include:

- download failures
- extraction crashes
- Docling instability on individual PDFs
- timeout behavior
- low-quality Greek extraction requiring OCR
- skiplist filtering

The recovery goal is not only to continue. It is to continue in a way that remains analyzable later.

## Retention

Retention means deciding what should remain on disk after a run.

This is important because the current pipeline retains many artifacts:

- source files
- intermediate Markdown
- cleaned Markdown
- Docling JSON
- metrics JSON
- section outputs
- logs
- problematic or timeout outputs

This improves auditability and rerun flexibility, but increases storage footprint.

## Useful retention classes

For documentation and future design, artifacts are best thought of in four classes:

### 1. Authoritative state

Examples:

- metadata parquet
- stage status fields
- skiplist and explicit recovery markers

These should be preserved carefully.

### 2. Downstream-required outputs

Examples:

- Markdown needed by later stages
- section parquet outputs
- final exports

These are often required either for downstream processing or for the final product.

### 3. Rebuildable intermediates

Examples:

- some JSON byproducts
- temporary per-stage outputs that can be regenerated from authoritative state plus upstream artifacts

These may not need indefinite retention.

### 4. Debug and incident evidence

Examples:

- logs
- problematic file quarantines
- timeout directories

These are especially useful for failed runs or regression analysis.

## Why this matters now

Storage pressure is a real concern in long-running corpus work.

Without a clear retention model, the pipeline tends to accumulate everything indefinitely. That is convenient in the short term but costly in the long term.

## Current architectural tension

The current system benefits from heavy artifact retention because it supports:

- reruns
- debugging
- auditing
- manual inspection

But the same design creates pressure when:

- corpora are large
- many runs are kept
- artifacts are duplicated across experiments
- storage migration is needed

## What future redesigns may want

Future improvements may try to preserve the logical contract while changing storage policy:

- stronger `document_id` semantics
- clearer separation between durable state and cache-like outputs
- more explicit retention classes
- better garbage-collection rules after successful runs

## Documentation guidance

Every stage doc should help answer:

- what must be kept to resume from this point
- what is optional to keep
- what is only for debugging
- what can be safely regenerated

That makes documentation useful not only for current operators, but also for future architecture work.
