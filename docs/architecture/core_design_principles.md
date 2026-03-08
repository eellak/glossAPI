# Core Design Principles

This document explains why the pipeline looks the way it does.

## 1. Throughput matters, but only if the run remains recoverable

GlossAPI uses Docling because it offers high-throughput extraction, rich layout handling, and integration paths for OCR and enrichment.

However, throughput is not pursued blindly. The pipeline is designed to retain progress and isolate problematic documents when instability appears.

Practical consequence:

- batching and parallelism are useful
- recovery and resumability are mandatory

## 2. Text extraction success is not the same as text quality success

A PDF can produce output without producing usable output.

This matters especially for Greek corpora because many files contain broken or absent `ToUnicode` mappings. As a result:

- extraction may complete
- Markdown may be non-empty
- the text may still be unusable Greek

GlossAPI therefore treats text validation as a separate stage concern, not as an optional afterthought.

## 3. OCR is a corrective stage, not the default for every file

OCR is more expensive than text-layer extraction. Running OCR on everything would increase compute cost, runtime, and failure surface.

The pipeline instead tries to:

1. extract cheaply where possible
2. detect low-quality output
3. reroute only bad files through OCR

This selective approach is central to the current design.

## 4. Metadata is a first-class product of the pipeline

The pipeline does not only produce Markdown. It also produces run metadata describing:

- original acquisition context
- extraction outcomes
- quality metrics
- OCR needs and OCR success
- skip behavior
- stage progression

This metadata is essential for:

- resumability
- debugging
- corpus auditing
- post-run data analysis

## 5. Filesystem structure is part of the contract

Stage outputs are not arbitrary temporary files. The pipeline expects specific folders, naming rules, and metadata locations.

That means contributors should treat:

- artifact paths
- naming conventions
- parquet locations
- chunk suffix behavior

as contract-like behavior unless deliberately redesigned.

## 6. Intermediate artifacts are retained intentionally

Keeping intermediate artifacts increases storage usage, but it provides strong benefits:

- easier debugging
- easier re-entry into later stages
- auditability
- ability to inspect what changed between stages

This is a deliberate tradeoff, not just accidental accumulation.

## 7. Recovery behavior is part of normal operation

The pipeline assumes that some documents will:

- fail to download
- fail to parse
- time out
- produce unusable Greek
- crash Docling

So recovery is not a rare edge case. It is part of standard corpus processing.

## 8. Current implementation details should be distinguished from logical contracts

The current implementation relies heavily on:

- filenames and stems
- parquet files on local disk
- retained intermediate artifacts

Those are the current mechanics, but the deeper logical contract is:

- each document needs a stable identity
- each stage needs explicit inputs and outputs
- each stage should emit usable status and diagnostics
- reruns should be able to distinguish success, failure, skip, and retry states

This distinction matters for future redesign work around storage, streaming, or chunk-level execution.
