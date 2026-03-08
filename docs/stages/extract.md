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
