# Docling Throughput and Batching

Docling is the central extraction engine for GlossAPI when throughput, OCR integration, or richer layout processing is required.

## Why Docling is used

Docling provides several advantages that matter for this pipeline:

- good extraction throughput
- support for PDF layout processing
- integration with OCR backends
- support paths for formula and code enrichment
- better fit for heterogeneous academic corpora than a purely minimal extractor

In GlossAPI, Docling is not just a library dependency. It is one of the main architectural choices.

## Backend model in GlossAPI

Phase 1 extraction distinguishes between:

- a safer backend path
- a Docling-oriented backend path

The backend is resolved in the extraction phase based on requested behavior. In practice:

- safe extraction is preferred when no heavy OCR or enrichment path is required
- Docling is selected when OCR or enrichment is needed
- Docling may also be chosen when throughput is prioritized and the environment is stable enough

This is why the pipeline keeps both performance-oriented and defensive extraction behavior available.

## Batching and parallelism

The codebase is structured to support high-throughput extraction through:

- batch policy configuration
- thread configuration
- accelerator selection
- GPU-aware OCR integration when needed

Batching helps because setup overhead can dominate for small files, and corpora often contain many similar documents.

Parallelism helps because extraction is a corpus task, not usually a single-document task.

## Why batching is not a free win

In theory, increasing batch size should improve throughput. In practice, Docling can become unstable, especially with more than one file in flight under some configurations.

That means batching has to be understood as a controlled optimization, not as an unconditional default.

The operational question is not:

- "Can Docling batch?"

It is:

- "Can Docling batch in this environment, on this corpus, with recoverable failure semantics?"

## Throughput vs stability tradeoff

The project currently balances two goals:

1. use Docling to get practical throughput
2. isolate bad documents and preserve progress when Docling misbehaves

This is a key design tradeoff. Contributors should resist oversimplifying it in either direction:

- purely maximizing batching can make runs fragile
- purely disabling batching can make large corpora impractically slow

## What should be documented for operators

Operators need to understand:

- which backend they are using
- whether OCR or enrichment forces Docling behavior
- whether batching is enabled
- how to recognize when the throughput optimization is causing instability
- what artifacts and metadata remain trustworthy after a partial failure

## What should be documented for contributors

Contributors need to understand:

- where backend selection happens
- where batch policy is configured
- what assumptions exist around one-file vs multi-file processing
- how failures should degrade
- how output and metadata contracts must survive recovery

## Future design pressure

Docling throughput behavior is also connected to future architectural changes:

- smaller storage footprint
- chunk-level execution
- lower operational fragility
- clearer document identity beyond filename stems

For that reason, throughput policy should be documented as an operational strategy, not as an invisible implementation detail.
