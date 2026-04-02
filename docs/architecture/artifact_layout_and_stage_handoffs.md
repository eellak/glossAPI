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

For DeepSeek OCR, there is an important distinction between execution-time shards and stage handoff artifacts:

- Multi-GPU `exact_fill` may execute shards such as `doc__p00001-00096` internally to keep GPU lanes full.
- Those shard names are operational artifacts, not the downstream contract for OCR outputs.
- After worker completion, the runner reassembles canonical `markdown/<stem>.md` and `json/metrics/<stem>.metrics.json` files for each source PDF.
- If OCR started from canonical corpus metadata, the authoritative OCR handoff should also include a canonical parquet where corrected `text` is embedded back into the same document rows. Detached markdown alone is not the full stage handoff in that case.
- Canonical OCR markdown page boundaries are annotated with `<!-- page:N -->` comments next to the page-split marker, and the parser remains backward-compatible with legacy unnumbered separators.
- Original shard markdown and shard metrics are moved under `sidecars/ocr_shards/` for debugging and audit trails.
- If a repair retry trips the garbage cutoff again, the canonical markdown keeps the page slot but blanks the page content rather than preserving the bad first-pass OCR.

For multi-GPU vLLM OCR, there is now a second class of operational artifacts under `sidecars/ocr_runtime/`:

- `work_queue.sqlite`: durable batch queue state for the current OCR run
- `worker_*.runtime.json`: per-worker heartbeat and timing state
- `gpu_preflight.json`: GPU readiness checks such as persistence mode
- `gpu_telemetry.jsonl`: sampled GPU utilization and process telemetry
- `runtime_summary.json`: queue completion state plus steady-state timing windows

The runtime queue now has two phases inside the same operational state:

- first-pass shard batches
- repair shard batches published after first pass completes

Repair queue durability and repair execution batching are intentionally separate concerns:

- the durable queue records individual repair work items so retries, failure accounting, and resume logic stay precise
- workers may pack multiple pending repair items into one larger execution batch to keep GPUs busy during the repair tail

These runtime artifacts are operational state, not downstream stage inputs. They are intended for monitoring, debugging, and safe resumption logic.

Downstream stages should therefore consume canonical OCR outputs, not shard artifacts.

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
