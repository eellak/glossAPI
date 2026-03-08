# Section Stage

## Purpose

The section stage splits Markdown documents into structural units suitable for classification and downstream analysis.

## Main responsibilities

- select eligible documents for sectioning
- parse Markdown structure
- emit section-level parquet data

## Main inputs

- Markdown artifacts
- metadata indicating which files are acceptable for downstream processing

## Main outputs

- section parquet outputs under `sections/`

## Selection semantics

This stage should prefer documents that are considered usable, either because they passed quality checks or because OCR repaired them sufficiently.

That means sectioning is downstream of quality gating, not independent from it.

## Failure concerns

Typical issues include:

- no usable markdown available
- metadata selection mismatches
- malformed markdown structure

## Contributor note

If section selection rules change, the documentation must also explain how eligibility is determined from metadata and artifacts.
