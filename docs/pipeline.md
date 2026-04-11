# Pipeline Overview & Artifacts

GlossAPI is a staged pipeline. You can enter at any stage and use the same folder for input and output.

## Corpus usage contract

The `Corpus` class is the stable surface of the project. New functionality should plug into the existing phase mixins so callers can stick to the small set of entrypoints (`download()`, `extract()`, `clean()`, `ocr()`, `section()`, `annotate()`, `export/jsonl*()`). The expected usage pattern is a short script that chains these calls; avoid ad-hoc monkeypatches or bypassing the orchestrator when adding features so downstream users retain resumability and consistent artifacts.

## Stage Map

| Stage | Main code | Typical inputs | Important parameters | Main outputs |
| --- | --- | --- | --- | --- |
| Download | `Corpus.download()`, `GlossDownloader.download_files()` | metadata parquet with a URL column | `input_parquet`, `links_column`, `parallelize_by`, downloader kwargs | `downloads/`, `download_results/*.parquet` |
| Extract (Phase‑1) | `Corpus.prime_extractor()`, `Corpus.extract()`, `GlossExtract.extract_path()` | files in `downloads/` or explicit paths | `input_format`, `phase1_backend`, `use_gpus`, `devices`, `workers_per_device`, `export_doc_json`, `emit_formula_index` | `markdown/<stem>.md`, `json/<stem>.docling.json(.zst)`, `json/metrics/*.json` |
| Clean | `Corpus.clean()` | `markdown/*.md` | `threshold`, `drop_bad`, `empty_char_threshold`, `empty_min_pages` | `clean_markdown/<stem>.md`, cleaner report parquet, parquet flags such as `filter` and `needs_ocr` |
| OCR retry | `Corpus.ocr(mode='ocr_bad'...)` | parquet rows flagged by cleaner | `mode`, `fix_bad`, `use_gpus`, `devices` | refreshed `markdown/<stem>.md`, refreshed cleaner/parquet metadata |
| Phase‑2 enrich | `Corpus.ocr(mode='math_only'...)`, `Corpus.formula_enrich_from_json()` | `json/<stem>.docling.json(.zst)` and optional formula index | `math_enhance`, `math_batch_size`, `math_dpi_base`, `targets_by_stem` | updated `markdown/<stem>.md`, `json/<stem>.latex_map.jsonl` |
| Section | `Corpus.section()`, `GlossSection.to_parquet()` | markdown selected by cleaner/parquet | no major public knobs | `sections/sections_for_annotation.parquet` |
| Annotate | `Corpus.annotate()`, `GlossSectionClassifier.classify_sections()`, `GlossSectionClassifier.fully_annotate()` | section parquet and classifier model | `annotation_type`, `fully_annotate` | `classified_sections.parquet`, `fully_annotated_sections.parquet` |
| Triage / export | `Corpus.triage_math()`, `Corpus.jsonl()` | metrics, parquet metadata, cleaned markdown | output path for JSONL | parquet routing hints, JSONL export |

## Stage Contracts

### 1. Download

- Main code: `Corpus.download()` -> `GlossDownloader.download_files()`
- Purpose: read a metadata parquet, expand list/JSON URL cells, deduplicate URLs, download supported file types, and checkpoint progress.
- Typical inputs:
  - a parquet file in `input_dir` or an explicit `input_parquet`
  - a URL column such as `url` or `links_column`
- Main outputs:
  - downloaded files in `downloads/`
  - partial/final results in `download_results/`
- Read this next if you want the scheduler details: `gloss_downloader.py`

### 2. Extract (Phase‑1)

- Main code: `Corpus.prime_extractor()`, `Corpus.extract()`, `GlossExtract.ensure_extractor()`, `GlossExtract.extract_path()`
- Purpose: convert source files to markdown and optional intermediate JSON artifacts.
- Typical inputs:
  - files already present in `downloads/`
  - or explicit `file_paths`
- Important parameters:
  - `phase1_backend='safe'|'docling'|'auto'`
  - `use_gpus='single'|'multi'`
  - `workers_per_device` to fan out more than one extraction worker onto each GPU
  - `export_doc_json` and `emit_formula_index` for later Phase‑2 work
- Operational note:
  - `force_ocr` is deprecated and ignored in Phase‑1; use `Corpus.ocr(backend='deepseek')` after `clean()` for OCR remediation
- Main outputs:
  - canonical markdown in `markdown/<stem>.md`
  - optional Docling JSON and index artifacts in `json/`
  - per-document and per-page metrics in `json/metrics/`

### 3. Clean

- Main code: `Corpus.clean()`
- Purpose: run the Rust cleaner, remove low-quality or noisy markdown,
  and mark documents that may need OCR retry before moving on.
- Typical inputs:
  - `markdown/*.md`
  - metadata parquet, if available
- Important parameters:
  - `threshold` and `drop_bad`
  - `empty_char_threshold` and `empty_min_pages` for OCR fallback decisions
- Main outputs:
  - cleaned markdown in `clean_markdown/`
  - updated parquet metadata with quality and OCR-related flags
- Runtime/debug artifacts:
  - `.processing_state.pkl` keeps track of progress so interrupted runs can resume
  - `problematic_files/` keeps files that could not be cleaned successfully
  - `timeout_files/` keeps files that exceeded the cleaning time limit

### 4. OCR Retry and Phase‑2 Enrichment

- Main code: `Corpus.ocr()` and `Corpus.formula_enrich_from_json()`
- Purpose:
  - rerun OCR only for documents marked bad by the cleaner
  - optionally decode formula/code regions from Docling JSON into markdown
- Architecture boundary:
  - corpus-side policy and parquet updates now live in `src/glossapi/corpus/ocr/`
  - runtime execution stays in `src/glossapi/ocr/deepseek/`
- Modes:
  - `ocr_bad`
  - `math_only`
  - `ocr_bad_then_math`
- Main outputs:
  - refreshed `markdown/<stem>.md`
  - `json/<stem>.latex_map.jsonl` when math/code enrichment runs
- Read this next if you are changing OCR internals:
  - `architecture/corpus_ocr_stack.md`
  - `architecture/deepseek_runner_stack.md`
  - `operations/deepseek_runtime_contract.md`
  - `operations/deepseek_single_gpu_benchmarking.md`

### 5. Section and Annotate

- Main code: `Corpus.section()`, `GlossSection.to_parquet()`, `Corpus.annotate()`, `GlossSectionClassifier.*`
- Purpose:
  - split markdown into sections suitable for classification
  - classify sections and optionally expand coarse labels into full document structure
- Main outputs:
  - `sections/sections_for_annotation.parquet`
  - `classified_sections.parquet`
  - `fully_annotated_sections.parquet`

## Artifact Layout

The tree below shows the main folders and files GlossAPI can create under
the output directory.

To make the layout easier to follow, artifacts are grouped by the role they
play in the pipeline:

- canonical — the main outputs a stage is expected to produce, and the
  files later stages usually depend on
- runtime — state files used to resume work safely if a run is interrupted
- debug — extra files kept around when something fails or needs a closer look

OUT/
├── downloads/                                  (canonical)
│   └── problematic_math/                       (debug)
├── download_results/                           (canonical)
├── markdown/                                   (canonical)
│   └── <stem>.md
├── clean_markdown/                             (canonical)
│   └── <stem>.md
├── json/                                       (canonical)
│   ├── <stem>.docling.json(.zst)
│   ├── <stem>.formula_index.jsonl
│   ├── <stem>.latex_map.jsonl
│   ├── metrics/
│   │   ├── <stem>.metrics.json
│   │   └── <stem>.per_page.metrics.json
│   └── problematic_math/                       (debug)
├── sections/                                   (canonical)
│   └── sections_for_annotation.parquet
├── classified_sections.parquet                 (canonical)
├── fully_annotated_sections.parquet            (canonical)
├── .processing_state.pkl                       (runtime)
├── problematic_files/                          (debug)
└── timeout_files/                              (debug)

Notes:
- Enriched Markdown replaces the plain Markdown (single canonical location).
- Metrics lived under `markdown/` in earlier versions; they now live under `json/metrics/`.
- When math enrichment cannot recover after the configured number of respawns, the corresponding PDFs and Docling artifacts are copied into the `problematic_math/` folders above and the stems are added to the fatal skip-list for later review.
- The same folder can act as both `input_dir` and `output_dir`; the pipeline creates its own subdirectories under that root.

## Readability Shortcut

If you only need the shortest path through the system:

1. `Corpus.download()` if you start from URLs.
2. `Corpus.extract()` for Phase‑1 markdown.
3. `Corpus.clean()` to decide what needs OCR.
4. `Corpus.ocr()` for selective OCR and optional math/code enrichment.
5. `Corpus.section()` and `Corpus.annotate()` for structured outputs.

If you need to jump from these ideas to the source files, see `code_map.md`.

## Exporting corpora

Use `Corpus.jsonl(...)` when you want a single JSONL file (e.g. quick inspection) and `Corpus.jsonl_sharded(...)` when preparing pretraining releases. Both calls accept the same knobs for renaming the text column, nesting pipeline metadata, and wiring an external source-metadata parquet.

```python
from pathlib import Path
from glossapi import Corpus

corpus = Corpus(input_dir=Path("input"), output_dir=Path("out"))

shards = corpus.jsonl_sharded(
    Path("out/export"),
    shard_size_bytes=500 * 1024 * 1024,
    shard_prefix="train",
    text_key="text",
    metadata_key="pipeline_metadata",
    metadata_fields=[
        "filter",
        "greek_badness_score",
        "is_empty",
        "latin_percentage",
        "mojibake_badness_score",
        "needs_ocr",
        "percentage_greek",
        "polytonic_ratio",
    ],
    include_remaining_metadata=False,
    metadata_path=Path("out/download_results/didaktorika_downloads_enhanced.parquet"),
    source_metadata_key="source_metadata",
    source_metadata_fields=["filename", "language", "handle_url", "date_accepted"],
    source_metadata_path=Path("out/source_metadata/didaktorika_full_enriched_FINAL.parquet"),
)
```

### Loading in downstream trainers

Hugging Face Datasets can stream the resulting `.jsonl.zst` shards without unpacking:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="out/export/train-*.jsonl.zst",
    streaming=True,
)["train"]

for row in dataset:
    text = row["text"]  # pipeline metadata is under row["pipeline_metadata"]
    break
```

For analytics or training-time filtering, keep the Parquet sidecars keyed by `doc_id` (or filename) and use PyArrow predicates:

```python
import pyarrow.dataset as ds

dataset = ds.dataset("out/metadata/source_metadata.parquet", format="parquet")
recent_greek = dataset.to_table(
    filter=(ds.field("language") == "Ελληνικά") &
            (ds.field("date_accepted") >= "2018-01-01")
)
```

These snippets are mirrored in the test suite so regressions in file layout or compression settings are caught automatically.
