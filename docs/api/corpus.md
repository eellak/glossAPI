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
- `downloader_config`: defaults for `download()` (e.g., concurrency, cookies)
- Main side effects: creates the standard output folders and lazy-initializes the extractor, sectioner, and classifier.

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
  skip_existing: bool = True,
  use_gpus: str = 'single',        # 'single'|'multi'
  devices: list[int] | None = None,
  use_cls: bool = False,
  benchmark_mode: bool = False,
  export_doc_json: bool = True,
  emit_formula_index: bool = False,
) -> None
```

- Purpose: Phase‑1 extraction from source files into markdown plus optional JSON intermediates.
- Typical inputs:
  - files already present in `downloads/`
  - or explicit `file_paths`
- Important parameters:
  - `phase1_backend='safe'|'docling'|'auto'`: PyPDFium for stability vs Docling for native layout/OCR
  - `force_ocr=True`: turn on OCR during extraction
  - `use_gpus='multi'`: use all visible GPUs through a shared work queue
  - `export_doc_json=True`: write `json/<stem>.docling.json(.zst)`
  - `emit_formula_index=True`: also write `json/<stem>.formula_index.jsonl`
- Main outputs:
  - `markdown/<stem>.md`
  - `json/<stem>.docling.json(.zst)` when enabled
  - `json/metrics/<stem>.metrics.json`
  - `json/metrics/<stem>.per_page.metrics.json`

## clean()

```python
clean(
  input_dir: str | Path | None = None,
  threshold: float = 0.10,
  num_threads: int | None = None,
  drop_bad: bool = True,
) -> None
```

- Purpose: run the Rust cleaner/noise pipeline and decide which documents are safe for downstream processing.
- Typical inputs:
  - `markdown/*.md`
  - metadata parquet if present
- Important parameters:
  - `threshold`: badness threshold
  - `drop_bad`: whether to remove bad files from downstream selection
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
  mode: str | None = None,
  device: str | None = None,
  model_dir: str | Path | None = None,
  max_pages: int | None = None,
  persist_engine: bool = True,
  limit: int | None = None,
  dpi: int | None = None,
  precision: str | None = None,
  math_enhance: bool = True,
  math_targets: dict[str, list[tuple[int,int]]] | None = None,
  math_batch_size: int = 8,
  math_dpi_base: int = 220,
  use_gpus: str = 'single',
  devices: list[int] | None = None,
  force: bool | None = None,
) -> None
```

- Purpose: selective OCR retry and optional Phase‑2 math/code enrichment.
- Mode selection:
  - `ocr_bad`: rerun OCR only for cleaner-flagged docs
  - `math_only`: run enrichment from existing Docling JSON
  - `ocr_bad_then_math`: OCR flagged docs, then enrich them
- Important parameters:
  - `mode`, `fix_bad`, `math_enhance`
  - `use_gpus`, `devices`
  - `math_targets` to restrict enrichment to specific items
- Main outputs:
  - refreshed `markdown/<stem>.md`
  - refreshed cleaner/parquet metadata after OCR reruns
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

- Purpose: Phase‑2 GPU enrichment from previously exported Docling JSON.
- Typical inputs:
  - `json/<stem>.docling.json(.zst)`
  - optional formula/code index data
- Important parameters:
  - `files`: restrict to specific stems
  - `device`, `batch_size`, `dpi_base`
  - `targets_by_stem`: target specific `(page_no, item_index)` tuples
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
  input_parquet: str | Path,
  *,
  links_column: str | None = None,
  parallelize_by: str | None = None,
  verbose: bool | None = None,
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
  - downloader kwargs via `**kwargs` for concurrency, SSL, cookies, retries, checkpoints, etc.
- Main outputs:
  - downloaded files in `downloads/`
  - partial/final results in `download_results/`
  - returned `pd.DataFrame` with download status and metadata

## triage_math()

- Purpose: summarize per-page metrics and recommend Phase‑2 for math-dense docs.
- Inputs: `json/metrics/<stem>.per_page.metrics.json`
- Outputs: updated `download_results` parquet with routing fields such as formula totals and phase recommendation

## Suggested Reading Order

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
