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
