# API Reference — `glossapi.Corpus`

The `Corpus` class is the high‑level entrypoint for the pipeline. Below are the most commonly used methods.

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

- Phase‑1 extraction; set `force_ocr=True` for OCR.
- Docling layout JSON now writes by default (`json/<stem>.docling.json(.zst)`); set `emit_formula_index=True` to also produce `json/<stem>.formula_index.jsonl`.
- Set `use_gpus='multi'` to use all visible GPUs (shared queue).

## clean()

```python
clean(
  input_dir: str | Path | None = None,
  threshold: float = 0.10,
  num_threads: int | None = None,
  drop_bad: bool = True,
) -> None
```

- Runs the Rust cleaner/noise metrics and populates parquet with badness; sets `good_files` and points `markdown_dir` to cleaned files for downstream.

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

- Convenience shim that re‑runs `extract(force_ocr=True)` on cleaner-flagged documents and, by default, performs math/code enrichment unless `math_enhance=False`.

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

- Phase‑2 enrichment from JSON. Writes enriched MD into `markdown/<stem>.md`, and `json/<stem>.latex_map.jsonl`.

## section(), annotate()

```python
section() -> None
annotate(annotation_type: str = 'text', fully_annotate: bool = True) -> None
```

- `section()` builds `sections/sections_for_annotation.parquet`.
- `annotate()` classifies sections and saves `classified_sections.parquet` and `fully_annotated_sections.parquet`.

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

- Concurrent downloader with per‑domain scheduler, retries, and checkpoints. Outputs `download_results/*.parquet` and files in `downloads/`.

## triage_math()

Summarizes per‑page metrics and recommends Phase‑2 for math‑dense docs. Updates `download_results` parquet.

---

See also:
- Configuration and environment variables: ../configuration.md
- OCR and math enrichment details: ../ocr_and_math_enhancement.md
