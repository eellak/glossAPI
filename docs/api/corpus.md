# API Reference — `glossapi.Corpus`

The `Corpus` class is the high‑level entrypoint for the pipeline. Below are the most commonly used methods.

Use this page as a compact contract reference. For the stage-by-stage artifact view, see `../pipeline.md`. For the source-level ownership map, see `../code_map.md`.

## Constructor

```python
glossapi.Corpus(
  input_dir: str | Path,
  output_dir: str | Path,
  section_classifier_model_path: str | Path | None = None,
  extraction_model_path: str | Path | None = None,
  metadata_path: str | Path | None = None,
  annotation_mapping: dict[str, str] | None = None,
  downloader_config: dict[str, Any] | None = None,
  log_level: int = logging.INFO,
  verbose: bool = False,
)
```

- `input_dir`: source files (PDF/DOCX/HTML/…)
- `output_dir`: pipeline outputs (markdown, json, sections, …)
- `downloader_config`: defaults for `download()` (e.g., concurrency, retries; public targets only, no cookies or credentials)
- Main side effects: creates the standard output folders and lazy-initializes the extractor, sectioner, and classifier.

## run()

```python
run(
  files=(),                      # local paths or URLs; empty = every document the corpus already knows
  *,
  stages=('extract', 'clean', 'sections', 'export'),
  workers: dict | None = None,
  settings: dict | None = None,  # e.g. {'debug': 'off', 'memory_saving': 'normal'}
  resume: str | None = None,     # a run_id to resume
  retry_failed: bool = False,
  plan: dict | None = None,      # a saved `glossapi plan --json` body
) -> dict
```

- Purpose: one streaming run of the selected stages, recorded in the corpus ledger `corpus.sqlite3`. Each stage's
  prerequisites are added (for example `export` adds `extract` and `clean`). `glossapi run` calls the same code.
- Returns the run data (`run_id`, `state` and counts), the same object as `glossapi run --json` prints in `data`.
- Stage workers are separate processes started with `spawn`. They import the calling script again, so guard a script
  that calls `run()` with `if __name__ == "__main__":`; without the guard the run fails in a confusing way.
- `resume=RUN_ID` restores that run's frozen plan and settings (no other arguments may change them). Interrupted or
  lost attempts are retried automatically, at most 3 attempts per task; `retry_failed=True` retries the failed tasks
  of the resumed run. See [Resumability, recovery and retention](../architecture/resumability_recovery_and_retention.md).

## extract()

```python
extract(
  input_format: str = 'all',
  num_threads: int | None = None,
  accel_type: str = 'CUDA',        # 'CPU'|'CUDA'|'MPS'|'Auto'
  *,
  force_ocr: bool = False,
  formula_enrichment: bool = False,
  code_enrichment: bool = False,
  filenames: list[str] | None = None,
  file_paths: list[str | Path] | None = None,
  skip_existing: bool = True,
  use_gpus: str = 'single',        # 'single'|'multi'
  devices: list[int] | None = None,
  use_cls: bool = False,
  benchmark_mode: bool = False,
  export_doc_json: bool = True,
  emit_formula_index: bool = False,
  phase1_backend: str = 'auto',    # 'auto'|'safe'|'docling'
  repair_fonts: bool = True,
  workers_per_device: int = 1,
  batched: bool | None = None,
  page_range: tuple[int, int] | None = None,
  ocr_options = None,
) -> None
```

- Purpose: Phase‑1 extraction from source files into markdown plus optional JSON intermediates (Docling backend only).
- Typical inputs:
  - files already present in `downloads/`
  - or explicit `file_paths`
- Important parameters:
  - `phase1_backend='safe'|'docling'|'auto'`: PyPDFium for stability vs Docling for native layout extraction; `'auto'` (default) uses the safe backend unless formula or code enrichment is requested
  - `repair_fonts=True` (default): repair eligible ToUnicode maps into `repaired_pdfs/<stem>.pdf` before extraction; each call repairs again from the source
  - `batched=None` (default): cross-document layout batching only for single-device, layout-only Docling on CUDA; CPU, MPS and the safe backend stay sequential
  - `page_range=(first, last)`: inclusive one-based pages, explicit sequential single-device Docling only
  - `ocr_options`: an external Docling OCR engine's options; sequential single-device Docling only
  - `force_ocr`: deprecated no-op kept for compatibility; OCR remediation now lives in `Corpus.ocr(mode='ocr_bad')`
  - `use_gpus='multi'`: use all visible GPUs through a shared work queue
  - `workers_per_device`: fan out more than one extraction worker onto a single visible GPU when measuring throughput
  - `GLOSSAPI_DOCLING_MAX_BATCH_FILES`: optional environment override for how many PDFs one Docling worker processes per extractor batch; GlossAPI keeps the default at `1` until a benchmark proves a larger batch is safe on the target node
  - `GLOSSAPI_DOCLING_BATCH_TARGET_PAGES`: optional environment override for the page budget of each queued multi-GPU Docling work item; use it with benchmark checkpoints when long PDFs dominate the tail
  - `GLOSSAPI_DOCLING_PAGE_BATCH_SIZE`: optional environment override for Docling's internal `settings.perf.page_batch_size`; use it when a GPU can hold more pages in flight than the default internal batch window
  - `export_doc_json=True`: write `json/<stem>.docling.json(.zst)` (Docling backend only; the safe backend writes none)
  - `emit_formula_index=True`: also write `json/<stem>.formula_index.jsonl` (Docling backend only)
- Main outputs:
  - `markdown/<stem>.md`
  - `json/<stem>.docling.json(.zst)` when enabled with `phase1_backend='docling'`
  - `json/metrics/<stem>.metrics.json`
  - `json/metrics/<stem>.per_page.metrics.json`

## clean()

```python
clean(
  input_dir: str | Path | None = None,
  threshold: float = <unset>,      # deprecated, no effect
  num_threads: int | None = None,
  drop_bad: bool = True,
  *,
  write_cleaned_files: bool = True,
  ocr_model_dir = None,            # deprecated, no effect
  force_ocr_fallback: bool = False,  # deprecated, no effect
  empty_char_threshold: int = 0,
  empty_min_pages: int = 0,
) -> None
```

- Purpose: run the Rust cleaner/noise pipeline and decide which documents are safe for downstream processing.
- Typical inputs:
  - `markdown/*.md`
  - metadata parquet if present
- Important parameters:
  - `threshold` (deprecated, no effect): passing it raises a `DeprecationWarning`. Selection uses fixed gates: a
    document is rejected when its `mojibake_badness_score` is above 0.1 or its Greek badness score is above 60, and
    documents flagged `needs_ocr` are not selected.
  - `drop_bad=True` removes the rejected documents from downstream processing; `False` keeps every file and only
    records the scores
  - `write_cleaned_files=False` skips writing cleaned Markdown; metrics and parquet updates still happen
  - `empty_char_threshold`, `empty_min_pages`: heuristics for OCR rerun recommendation
- Main outputs:
  - `clean_markdown/<stem>.md`
  - cleaner report parquet
  - updated parquet columns such as `filter`, `needs_ocr`, and metrics fields
- Operational note: this stage is the quality gate that drives `section()` and `ocr()`.

## ocr()

```python
ocr(
  *,
  fix_bad: bool = True,
  page_routing: str | None = None, # None = "all" for the OCR modes, "flagged" for math_only
  mode: str | None = None,
  backend: str = "docling",
  ocr_options = None,
  device: str | None = None,
  max_pages: int | None = None,
  limit: int | None = None,        # accepted, no effect
  dpi: int | None = None,          # accepted, no effect
  math_enhance: bool = True,
  math_targets: dict[str, list[tuple[int,int]]] | None = None,
  math_batch_size: int = 8,
  math_dpi_base: int = 220,
  use_gpus: str = 'single',
  devices: list[int] | None = None,
  force: bool | None = None,       # deprecated alias of fix_bad
  reprocess_completed: bool | None = None,
  skip_existing: bool | None = None,  # deprecated; use reprocess_completed
  content_debug: bool = False,
) -> None
```

- Purpose: selective OCR retry and optional Phase‑2 math/code enrichment.
- Mode selection (without `mode`, `fix_bad` and `math_enhance` choose it: both → `ocr_bad_then_math`):
  - `ocr_bad`: OCR the selected pages (see `page_routing`)
  - `math_only`: run enrichment from existing Docling JSON; performs no OCR
  - `ocr_bad_then_math`: OCR first, then enrich the documents OCR did not touch from existing Docling JSON
- Page selection (`page_routing`):
  - default `"all"`, the page policy of `glossapi run`: every page that page quality routes is OCRed, in documents the
    cleaner flagged and in documents it did not
  - `"flagged"`: OCR only the documents the cleaner flagged (`needs_ocr`), the default before K4g
  - `mode="math_only"` performs no OCR and by default still skips only cleaner-flagged documents
  - a cleaner-flagged document whose page quality routed no page is not OCRed, in both settings, as `glossapi run`
    keeps its extracted text (K5 quality policy P4d); a document without a page-quality report keeps document-level
    routing and the OCR stage's own page quality picks its pages
- Important parameters:
  - `mode`, `fix_bad`, `page_routing`, `math_enhance`
  - `backend` must be `"docling"`; `ocr_options` selects a configured external plugin, otherwise the pinned Greek
    OCR plugin is used
  - `use_gpus`, `devices`
  - `math_targets` to restrict enrichment to specific items inside the named stems (other discovered stems are still
    enriched; use `formula_enrich_from_json(files=..., targets_by_stem=...)` to restrict documents too)
  - `limit` and `dpi` are accepted for compatibility and ignored
- Main outputs:
  - refreshed `markdown/<stem>.md`
  - refreshed cleaner/parquet metadata after OCR reruns
  - when metadata parquet is present, OCR now preserves the same row identity and embeds corrected `text` plus direct OCR sidecar pointers such as `ocr_markdown_relpath`, `ocr_metrics_relpath`, and `ocr_text_sha256`
  - `json/<stem>.latex_map.jsonl` when enrichment runs

## formula_enrich_from_json()

```python
formula_enrich_from_json(
  files: list[str] | None = None,
  *,
  device: str = 'cuda',
  batch_size: int = 8,
  dpi_base: int = 220,
  targets_by_stem: dict[str, list[tuple[int,int]]] | None = None,
) -> None
```

- Purpose: Phase‑2 GPU enrichment from previously exported Docling JSON (an explicit `phase1_backend='docling'`
  extraction; needs the `layout` package). It creates no JSON: without Docling JSON it logs that nothing was found
  and returns.
- Typical inputs:
  - `json/<stem>.docling.json(.zst)`
  - optional formula/code index data
- Important parameters:
  - `files`: restrict to specific stems; `None` (default) discovers every `json/<stem>.docling.json(.zst)`
  - `device`, `batch_size`, `dpi_base`
  - `targets_by_stem`: target specific `(page_no, item_index)` tuples within the named stems; pass the same stems as
    `files` to restrict the documents as well
- Main outputs:
  - enriched markdown back into `markdown/<stem>.md`
  - `json/<stem>.latex_map.jsonl`

## section(), annotate()

```python
section() -> None
annotate(annotation_type: str = 'text', fully_annotate: bool = True) -> None
```

- `section()`:
  - purpose: convert markdown into one row per section with structural flags
  - inputs: markdown selected by cleaner/parquet metadata
  - outputs: `sections/sections_for_annotation.parquet`
- `annotate()`:
  - purpose: classify sections and optionally expand them into full document structure
  - important parameters: `annotation_type='text'|'chapter'|'auto'`, `fully_annotate`
  - outputs: `classified_sections.parquet` and `fully_annotated_sections.parquet`

## download()

```python
download(
  input_parquet: str | Path | None = None,
  url_column: str = 'url',
  verbose: bool | None = None,
  *,
  parallelize_by: str | None = None,
  links_column: str | None = None,
  **kwargs,
) -> pd.DataFrame
```

- Purpose: fetch source files described in a parquet dataset.
- Typical inputs:
  - an explicit `input_parquet`
  - or the first parquet file found in `input_dir`
- Important parameters:
  - `links_column`: override URL column name
  - `parallelize_by`: choose grouping for the scheduler
  - `download_mode`: one of `standard`, `auto`, or `browser`
  - `browser_mode=True`: alias for `download_mode="browser"`
  - `download_policy_file`: route specific domains/URL patterns to `standard`, `auto`, or `browser`
  - downloader kwargs via `**kwargs` for concurrency, SSL, retries, checkpoints, etc. Public targets only: a URL with
    userinfo or a credential query field, or a `domain_cookies`/`credentials` setting, is refused with
    `credentials_unsupported` before any request.
- Main outputs:
  - downloaded files in `downloads/`
  - partial/final results in `download_results/`
  - returned `pd.DataFrame` with download status and metadata
- Resume: a rerun over the same input skips the URLs that already succeeded. The pending URLs go to the downloader
  with every input column and the provenance columns, as on the first run, so a resumed row keeps its source fields and
  its preassigned `filename_base` (see [Download stage](../stages/download.md)).

Browser-capable download mode is intended for browser-gated file endpoints where a real file still exists behind session/bootstrap checks. It is not a general viewer extractor. Viewer-only sources should still fail cleanly with a recorded error and no local file artifact.

Example:

```python
corpus.download(
    input_parquet="input_urls.parquet",
    download_mode="browser",
)
```

Policy-routed example:

```python
corpus.download(
    input_parquet="input_urls.parquet",
    download_policy_file="download_policy.yml",
)
```

## triage_math()

- Purpose: summarize per-page metrics and recommend Phase‑2 for math-dense docs.
- Inputs: `json/metrics/<stem>.per_page.metrics.json`
- Outputs: updated `download_results` parquet with routing fields such as formula totals and phase recommendation

## Suggested Reading Order

0. `run()` (or `glossapi run`) for a whole recorded run of the stages below.
1. `download()` if you start from URLs.
2. `extract()` for Phase‑1 layout/markdown.
3. `clean()` to decide what needs OCR.
4. `ocr()` if you need OCR retry or Phase‑2 enrichment.
5. `section()` and `annotate()` for structured downstream outputs.

---

See also:
- Code map: ../code_map.md
- Pipeline overview and artifacts: ../pipeline.md
- Configuration and environment variables: ../configuration.md
- OCR and math enrichment details: ../ocr_and_math_enhancement.md
