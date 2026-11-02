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

`Corpus.section()` sections the documents `Corpus.clean()` selected in the same `Corpus` object (`good_files`).
Without that list it selects from the metadata parquet: rows whose `filter` is `ok` or empty, or whose `ocr_success`
is true (and, if none match, rows whose legacy `extraction` column is `good`). If nothing is selected, it falls back
to every Markdown file and records `section_legacy_all_fallback`. In a streaming run only the run's planned
documents are sectioned.

That means sectioning is downstream of quality gating, not independent from it.

## Global section IDs

The numeric section `id` values in `sections_for_annotation.parquet` follow the order in which the filesystem lists
`markdown/` (`os.listdir`, not sorted). A streaming run records that listing in the ledger as an input of the
sections publication, so the IDs of a run can be traced, but the same files can get different IDs on a filesystem
that lists them in another order. Treat `id` as a compatibility field; the ledger keeps stable private section keys.

## Failure concerns

Typical issues include:

- no usable markdown available
- metadata selection mismatches
- malformed markdown structure

## Contributor note

If section selection rules change, the documentation must also explain how eligibility is determined from metadata and artifacts.
