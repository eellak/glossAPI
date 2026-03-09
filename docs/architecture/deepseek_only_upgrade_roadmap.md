# DeepSeek-Only Upgrade Roadmap

This document describes the planned migration from a mixed OCR stack to a simpler pipeline that keeps Docling for extraction and structure, but uses DeepSeek as the only OCR backend.

## Current status

As of March 9, 2026, the following work has already been completed:

- DeepSeek is the only supported OCR remediation backend in the pipeline
- stub execution is rejected for real OCR runs
- the dedicated DeepSeek runtime is managed through the uv-based setup flow
- RapidOCR implementation files and install profile have been removed
- real extract -> clean/evaluate -> OCR -> section validation has been run on capped Pergamos samples
- OCR progress artifacts were moved out of the canonical `markdown/` tree so downstream stages no longer treat them as real documents

The following work is intentionally not part of the completed set yet:

- Docling dependency upgrades
- page-level OCR reevaluation experiments
- broader corpus-level comparative benchmarking beyond the capped validation runs

## Remaining TODO to wrap up the implemented changes

These are the remaining tasks for closing out the already-implemented migration work:

1. review and curate the final commit contents
2. keep only source, docs, and test changes that belong in the `development` branch
3. exclude local artifacts, downloaded models, disposable environments, and ad hoc validation output from the commit
4. optionally run one more small real-PDF compatibility slice if an extra release-confidence check is desired
5. create or switch to the `development` branch and push the finalized change set there

This means the migration implementation itself is effectively done; what remains is mainly release hygiene and branch preparation.

## Target architecture

The target shape is:

1. `download()`
2. `extract()` via safe backend or Docling
3. `clean()` and compute Greek-quality routing
4. `ocr()` via DeepSeek only for documents that need remediation
5. `section()`
6. `annotate()`
7. `export()`

Important boundary:

- keep `Docling` for extraction, layout, Markdown, JSON artifacts, and optional formula/code enrichment
- remove `RapidOCR` from the OCR path and installation surface
- enforce `GLOSSAPI_DEEPSEEK_ALLOW_STUB=0` for production and release validation

This is a simplification, not a redesign of the entire pipeline contract.

## Why this direction

The current mixed OCR surface adds complexity in three places:

- dependency installation and CUDA compatibility
- runtime branching and operational support burden
- validation burden when one OCR path succeeds and another fails differently

The simplified design still preserves the important current properties:

- selective OCR after Greek-quality validation
- Docling-generated layout and JSON artifacts for downstream stages
- explicit operational metadata and rerun semantics

## Stage 1: DeepSeek-only OCR

Goal:

- make DeepSeek the only OCR remediation backend
- remove silent stub fallback from production paths

Changes:

- remove `rapidocr` as a supported OCR backend
- route `Corpus.ocr()` to DeepSeek only
- fail hard when DeepSeek runtime, weights, or CLI are unavailable
- keep the current document-level `needs_ocr` selection model

Do not change in this stage:

- Docling extraction contract
- sectioning and annotation behavior
- page-level routing policy
- formula/code enrichment policy

Why this stage exists:

- it gives the desired simplification without changing the rest of the pipeline contract at the same time
- it isolates OCR-engine risk from Docling-upgrade risk

Success criteria:

- no remaining production path imports or dispatches RapidOCR
- no final validation run succeeds via stub output
- documents flagged `needs_ocr=True` can still be remediated through DeepSeek

Status:

- completed

## Stage 2: Installation simplification

Goal:

- reduce the environment surface to what the simplified pipeline actually needs

Changes:

- remove the `rapidocr` install profile and `onnxruntime-gpu`
- simplify setup profiles around:
  - Docling extraction/runtime
  - DeepSeek OCR runtime
- remove unused requirement baggage where it is not imported by GlossAPI itself
- make Python version constraints match current upstream reality

Current constraint to fix:

- GlossAPI currently declares `requires-python = ">=3.8"` while current Docling requires Python `>=3.10`

Do not change in this stage:

- pipeline behavior
- artifact layout
- OCR routing logic

Why this stage exists:

- environment simplification should follow architectural simplification
- it is easier to reason about required packages once RapidOCR is gone

Success criteria:

- setup documentation exposes only the supported environments
- install instructions no longer mention removed OCR components
- Python floor and dependency pins are internally consistent

Status:

- completed for the currently supported DeepSeek-only flow
- final branch hygiene and commit curation still remain

## Stage 3: Docling upgrade

Goal:

- upgrade Docling after the OCR surface has already been simplified

Changes:

- update `docling`
- update `docling-core`
- update `docling-parse`
- update `docling-ibm-models`
- adapt any compatibility shims required by changed public APIs

Do not change in this stage:

- DeepSeek-only OCR decision
- page-level experiment
- formula/code enrichment policy unless explicitly validated

Why this stage exists:

- upgrading Docling before removing RapidOCR combines two unrelated breakage sources
- after Stage 1 the Docling integration surface is smaller and easier to validate

Success criteria:

- Phase-1 extraction still produces the documented canonical artifacts
- downstream sectioning, annotation, and export still consume the outputs
- metadata and resumability behavior do not regress

Status:

- deferred

## Stage 4: Re-evaluate retained Docling capabilities

Goal:

- decide which Docling-powered features remain justified after the simplification

Features to evaluate:

- formula enrichment
- code enrichment
- table structure extraction
- any extra model/artifact prefetch currently required for non-default functionality

Why this stage exists:

- some capabilities may still be valuable for technical corpora
- some may only be increasing runtime and failure surface

Rule:

- do not remove formula/code enrichment just because it simplifies the stack
- remove it only if real-corpus evaluation shows little or no value

Success criteria:

- every retained capability has a measurable purpose
- every removed capability has an explicit evaluation-based justification

Status:

- pending

## Stage 5: Page-level reevaluation experiment

Goal:

- test whether whole-document OCR reruns should be replaced or complemented by page-level escalation

Experiment shape:

- baseline branch: current document-level `needs_ocr` routing
- experiment branch: page-level or ROI-level routing

What stays fixed:

- DeepSeek remains the only OCR backend
- Docling remains the structured extraction/layout path

Why this is separate:

- it is an architectural experiment, not a prerequisite for the OCR simplification
- it should be compared against the stabilized DeepSeek-only baseline

Primary evaluation questions:

- does page-level escalation improve quality on long PDFs
- does it reduce OCR runtime and GPU cost
- does it preserve downstream sectioning and annotation quality

Status:

- pending

## Non-goals for the first pass

These are intentionally out of scope for the initial migration:

- replacing Docling JSON/layout artifacts with DeepSeek-native structured artifacts
- merging all runtime concerns into one universal environment regardless of ecosystem constraints
- changing artifact layout at the same time as OCR simplification
- treating synthetic, mocked, or stubbed tests as sufficient release validation

## Release sequence

The intended order is:

1. DeepSeek-only OCR and no-stub enforcement
2. installation simplification
3. Docling upgrade
4. retained-capability review
5. page-level experiment

This order keeps one major architectural assumption changing at a time.
