# Artifact Layout and Stage Handoffs

This document explains how the current pipeline uses on-disk artifacts to pass work from one stage to the next.

This is one of the most important implementation contracts in GlossAPI.

## Why this matters

The pipeline is not driven only by method calls. It is also driven by:

- directories
- filenames and stems
- parquet metadata
- presence or absence of stage artifacts

If these conventions change carelessly, stage chaining, reruns, and recovery can break.

## Core directories

The output directory typically contains these stage-relevant subdirectories:

- `downloads/`: downloaded source files
- `download_results/`: parquet metadata from acquisition and later updates
- `markdown/`: extracted Markdown
- `clean_markdown/`: cleaned Markdown
- `json/`: Docling and metrics byproducts
- `sections/`: section parquet outputs
- `logs/`: logs and stage-specific operational traces

These are not arbitrary folders. They define the current stage handoff model.

## Handoff pattern by stage

### Download -> Extract

`extract()` expects source documents to be available through the corpus download layout, typically under `downloads/`.

This means the downloader is responsible for more than fetching bytes. It is also preparing the corpus for the extraction contract.

### Extract -> Clean

`clean()` operates on Markdown outputs from extraction, typically in `markdown/`.

This is the point where extracted text becomes quality-assessable data.

### Clean -> OCR

`ocr()` does not simply scan markdown files and OCR everything. It consults metadata, especially OCR-related flags and quality signals, to decide which documents require remediation.

This means the handoff is:

- file artifacts from extraction
- metadata judgments from cleaning

### Clean/OCR -> Section

`section()` should operate on documents considered usable, either because:

- they passed quality checks
- they were successfully repaired through OCR

So this handoff depends on both Markdown availability and stage metadata.

### Section -> Annotate / Export

Later stages consume parquet outputs and related metadata rather than raw source files.

## Naming conventions

The pipeline uses filenames and normalized stems to reconnect outputs across directories.

This usually determines how the system finds relationships between:

- source files
- Markdown outputs
- JSON outputs
- metrics
- chunked outputs
- parquet metadata rows

This is practical, but it also means filenames act as part of document identity in the current design.

## Chunk-aware naming

Some processing modes produce chunked outputs, typically by encoding page ranges in filenames or stems.

That affects:

- artifact discovery
- metadata aggregation
- export-time reconstruction

Chunk suffix behavior is therefore part of the current contract.

## Authoritative state vs derived artifacts

Not every file has equal semantic importance.

The current system benefits from distinguishing:

- authoritative metadata state
- stage outputs needed downstream
- optional or debug-oriented artifacts
- rebuildable byproducts

This distinction becomes important for storage policy and future redesign.

## Operational risks in this model

The current on-disk contract is effective, but it creates risks:

- directory scanning can encode hidden assumptions
- filename-based identity can be fragile
- chunk naming can become overloaded
- retention of all intermediates can increase long-term storage costs

These are manageable as long as they are documented explicitly.

## Recommended contributor rule

If a change affects any of the following, treat it as a contract change and update docs:

- directory names
- file naming or stem normalization
- chunk naming
- which stage reads from which directory
- which artifacts are required for rerun or downstream stages
- where metadata parquet is expected to live
