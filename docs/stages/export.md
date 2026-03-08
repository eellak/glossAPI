# Export Stage

## Purpose

The export stage reshapes processed corpus data into downstream-friendly formats such as JSONL and aggregated metadata views.

## Main responsibilities

- read processed metadata and stage outputs
- aggregate chunked and unchunked document information
- emit export-oriented records for downstream consumers

## Main inputs

- metadata parquet
- section or text outputs depending on export mode
- chunk-aware naming and aggregation logic

## Main outputs

- exported JSONL or related downstream formats

## Why this stage matters

Export is where many earlier assumptions become visible:

- whether document identity is stable
- whether chunked outputs can be reconstructed correctly
- whether metadata remains coherent after reruns and recovery

## Contributor note

Changes to export logic often reveal hidden assumptions in the rest of the pipeline. Document those assumptions explicitly rather than leaving them embedded in naming or aggregation code.
