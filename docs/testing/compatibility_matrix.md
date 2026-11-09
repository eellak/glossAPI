# Compatibility And Regression Matrix

This document defines the release-validation matrix for the Greek OCR toolkit integration, Docling upgrades and other changes to the documented pipeline contracts. The current status of each gate is at the end of this page.

It is not a generic unit-test list. It is a contract-based validation plan tied to the documented pipeline behavior.

## Scope

This matrix applies to changes in:

- external OCR integration
- no-stub enforcement
- installation simplification
- Docling dependency upgrades
- page-level reevaluation experiments

## Validation policy

Release validation for these contracts must use:

- real PDFs
- real Docling
- real external OCR engine
- real GPUs where the code path requires them

Developer-only tests may still use mocks or lightweight stubs for fast iteration, but those do not satisfy release gates.

## Test levels

### L0: Install and import sanity

Purpose:

- prove the supported environments install cleanly and that removed components are truly gone

Typical inputs:

- fresh venv
- supported Python version

### L1: Lightweight smoke corpus

Purpose:

- prove the baseline end-to-end flow still works on the small repo corpus

Typical inputs:

- `samples/lightweight_pdf_corpus/`

### L2: Real-PDF contract validation

Purpose:

- prove the documented artifacts and metadata contracts still hold on real documents

Typical inputs:

- real PDFs from a representative sample

### L3: Multi-GPU and operational recovery

Multi-GPU OCR is a target that has not been qualified; the external OCR plugin runs on one device.

Purpose:

- prove the runtime behavior remains correct under parallel execution and rerun conditions

Typical inputs:

- multiple real PDFs
- at least two visible GPUs

### L4: Comparative corpus evaluation

Purpose:

- compare baseline and changed behavior on a real evaluation slice

Typical inputs:

- real corpus slice such as the Pergamos sample

## Mandatory invariants

The following must remain true unless a change explicitly revises the contract and updates the docs:

- canonical Markdown is written to `markdown/<stem>.md`
- Docling JSON artifacts are emitted when requested
- cleaner output still drives `needs_ocr`
- OCR remains selective: only pages that page quality routes (`page_routing="all"`, the default) or only documents the cleaner flagged (`page_routing="flagged"`)
- the ledger `corpus.sqlite3` is the durable record of a streaming run; the metadata parquet stays the compatibility record of the phase methods
- reruns skip completed work unless forced
- skiplist semantics remain explicit and stable
- no production path silently falls back to stub OCR

## Release-gate matrix

| ID | Level | Contract | Input | Run | Pass criteria | Negative assertions |
| --- | --- | --- | --- | --- | --- | --- |
| `ENV-001` | L0 | Python and packaging | Fresh environment | install supported profile(s) | install completes on supported Python floor | no reference to removed legacy OCR install modes |
| `EXT-001` | L1 | Safe Phase-1 extraction | lightweight corpus | `Corpus.extract(input_format="pdf")` | canonical Markdown produced | extraction must not depend on OCR extras |
| `EXT-002` | L2 | Docling Phase-1 extraction | real PDFs | `Corpus.extract(..., phase1_backend="docling", export_doc_json=True)` | Markdown, Docling JSON, metrics written to documented locations | artifact layout must not drift |
| `CLN-001` | L1/L2 | Cleaner metadata contract | extracted docs | `clean(drop_bad=False)` | metadata parquet updated with routing-relevant fields | no collapse of `needs_ocr` behavior |
| `MTH-001` | L2 | Formula/code enrichment compatibility | math-heavy real PDF | Docling extract plus Phase-2 enrichment | enriched outputs and metadata remain coherent | no schema drift breaking enrichment |
| `SEC-001` | L2 | Sectioning contract | usable real docs | `section()` | `sections/sections_for_annotation.parquet` produced | no empty-output regression caused by upstream changes |
| `ANN-001` | L2 | Annotation contract | section parquet | `annotate()` | classified outputs produced | model integration must not break on changed upstream text/layout |
| `EXP-001` | L2 | Export contract | processed docs | `jsonl()` / `jsonl_sharded()` | JSONL and metadata outputs match documented layout | no dropped metadata fields without explicit design change |
| `RES-001` | L3 | Resumability | interrupted or partial run | rerun with defaults | completed items skipped correctly | no duplicate reprocessing by default |
| `RES-002` | L3 | Force/reprocess semantics | prior successful run | rerun with force/reprocess flag | selected items are reprocessed | no stale completion flags blocking intended rerun |
| `SKP-001` | L3 | Skiplist semantics | run with known problematic items | extract/OCR rerun | skiplist excludes intended stems only | no hidden filtering of healthy items |
| `GPU-001` | L3 | Multi-GPU OCR (**target, not qualified**: the external plugin runs on one device today; see [OCR plugin migration](../ocr_plugin_migration.md)) | real PDF slice on 2 GPUs | external plugin OCR in parallel | work is distributed and completes per GPU | no worker success masking failures |
| `CMP-001` | L4 | Baseline quality comparison | Pergamos sample slice | compare pre/post change outputs | no material regression in artifact completeness and downstream usability | runtime improvement alone does not justify quality loss |
| `CMP-002` | L4 | Whole-text vs page-level experiment | long PDFs | compare baseline branch vs page-level branch | quality/runtime tradeoff explicitly measured | experimental branch does not replace baseline without evidence |

## Detailed test groups

### Install and runtime compatibility

What to prove:

- supported environment installs cleanly
- unsupported/removed OCR components are not required
- Python floor matches actual upstream dependencies

Critical checks:

- packaging metadata uses a supported Python minimum
- setup docs expose only supported install paths
- removal of the old OCR integration does not leave dead GlossAPI imports or entrypoints

## Extraction contract

What to prove:

- Phase-1 still produces canonical Markdown
- Docling extraction still produces JSON artifacts when requested
- metrics continue to be written where downstream stages expect them

Artifacts to check:

- `markdown/<stem>.md`
- `json/<stem>.docling.json(.zst)`
- `json/<stem>.formula_index.jsonl` when requested
- `json/metrics/<stem>.metrics.json`
- `json/metrics/<stem>.per_page.metrics.json`

## Cleaning and Greek-quality routing

What to prove:

- cleaner still computes routing decisions required for selective OCR
- Greek-text validation remains first-class rather than incidental cleanup

Fields to check in metadata parquet:

- `needs_ocr`
- `filter`
- Greek-quality and badness-related fields currently emitted by the cleaner

## external plugin OCR contract

What to prove:

- OCR remediation uses a configured external Docling engine
- no-stub enforcement is real
- recovered documents update metadata correctly

Required environment behavior:

- real model weights present
- real CLI/runtime path present

Negative checks:

- no markdown contains placeholder stub markers
- no OCR pass succeeds after an engine failure
- no removed OCR backend is referenced during final validation

## Formula and code enrichment

What to prove:

- if retained, enrichment still works with the upgraded Docling stack
- if later removed, the removal is justified by evaluation rather than convenience

Checks:

- enriched Markdown is generated where expected
- `json/<stem>.latex_map.jsonl` remains coherent when enrichment is enabled
- metadata updates for math enrichment still work

## Section, annotate, and export contracts

What to prove:

- downstream stages still consume the extraction outputs
- output layout and metadata structure remain compatible with the documented pipeline

Artifacts to check:

- `sections/sections_for_annotation.parquet`
- `classified_sections.parquet`
- `fully_annotated_sections.parquet`
- exported JSONL shards and related metadata

## Resumability and operational recovery

What to prove:

- reruns still honor completion state
- skiplist semantics remain intact
- multi-worker failures remain visible and recoverable

Checks:

- default rerun skips completed items
- explicit force/reprocess reruns the intended items
- problematic stems are persisted and not silently lost

## Comparative evaluation set

Suggested real-world slice:

- lightweight corpus for smoke validation
- representative real PDFs spanning:
  - short documents
  - medium documents
  - long documents
  - structure-rich documents
  - math-heavy documents where applicable

For current local evaluation work, a Pergamos sample manifest has been prepared outside the repo and can be used as the L3/L4 real-PDF slice.

## Gate order

For a change that touches these contracts, run gates in this order:

1. `ENV-*`
2. `EXT-*`
3. `CLN-*`
4. `OCR-*`
5. `MTH-*`
6. `SEC-*`, `ANN-*`, `EXP-*`
7. `RES-*`, `SKP-*`, `GPU-*`
8. `CMP-*`

This keeps low-level compatibility failures from being confused with downstream quality regressions.

## Current qualification status (0.2.0)

The external OCR migration is complete: the pinned Greek OCR toolkit is the only supported OCR engine, the removed
engines and install modes are gone, and Docling is an optional backend. The status of each gate group for 0.2.0:

| Gate | Status | Evidence |
| --- | --- | --- |
| `ENV-001` | Covered by tests; test required per target | Installer, declaration and capability tests run in the CPU suite. A fresh install must still be checked on each target platform and Python version. |
| `EXT-001`, `CLN-001`, `SEC-001`, `ANN-001`, `EXP-001` | Covered by the CPU test suite | Lightweight corpus and phase tests (`tests/test_corpus_phases_lightweight.py` and the stage tests); no GPU or Docling. |
| `EXT-002` | Optional backend; test required | Docling is mocked in the CPU suite. Run a real Docling extraction on the target before relying on Docling JSON. |
| `OCR-*` | CPU fixture covered; test required per target | The synthetic CPU fixture checks the routed plugin path; accelerator/device combinations must be qualified on the target machine (see [OCR plugin migration](../ocr_plugin_migration.md) and the CHANGELOG limitations). |
| `MTH-001` | Test required | Real Docling math inference is not part of the CPU suite; the phase API recipe needs a CUDA host with the `layout` package. Math is not executable in the streaming run yet. |
| `RES-*`, `SKP-*` | Covered by tests | The ledger, resume and retention suites. |
| `GPU-001` | Target, not qualified | The external plugin runs on one device; multi-GPU OCR has not been qualified. |
| `CMP-*` | Evaluation, not a release gate | Comparative runs on real slices are evaluation work, run when a change claims a quality or runtime effect. |
