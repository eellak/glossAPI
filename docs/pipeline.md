# Pipeline Overview & Artifacts

GlossAPI is a staged pipeline. You can enter at any stage and use the same folder for input and output.

## Corpus usage contract

The `Corpus` class is the stable surface of the project. New functionality should plug into the existing phase mixins so callers can stick to the small set of entrypoints (`run()`, `download()`, `extract()`, `clean()`, `ocr()`, `section()`, `annotate()`, `jsonl()`, `jsonl_sharded()`). The expected usage pattern is a short script that chains these calls, or one `Corpus.run()` / `glossapi run`; avoid ad-hoc monkeypatches or bypassing the orchestrator when adding features so downstream users retain resumability and consistent artifacts.

`Corpus.run()` (and `glossapi run`) executes the selected stages as one streaming run recorded in the corpus ledger
`corpus.sqlite3`; the ledger is what resume, retention, `glossapi status` and `glossapi report` read. Its stage
workers are `spawn` processes that import the calling script again, so guard a script that calls `Corpus.run()` with
`if __name__ == "__main__":`. See [Resumability, recovery and retention](architecture/resumability_recovery_and_retention.md).

## Stage Map

| Stage | Main code | Typical inputs | Important parameters | Main outputs |
| --- | --- | --- | --- | --- |
| Download | `Corpus.download()`, `GlossDownloader.download_files()` | metadata parquet with a URL column | `input_parquet`, `links_column`, `parallelize_by`, downloader kwargs | `downloads/`, `download_results/*.parquet` |
| Extract (Phase‑1) | `Corpus.prime_extractor()`, `Corpus.extract()`, `GlossExtract.extract_path()` | files in `downloads/` or explicit paths | `input_format`, `phase1_backend`, `repair_fonts=True`, `use_gpus`, `devices`, `workers_per_device`, `export_doc_json`, `emit_formula_index` | `markdown/<stem>.md`, `repaired_pdfs/<stem>.pdf`, `json/<stem>.docling.json(.zst)`, `json/metrics/*.json`; parquet columns `font_repair_status`, `font_repair_fonts`, `font_repair_valid_rate_before`, `font_repair_valid_rate_after` |
| Clean | `Corpus.clean()` | `markdown/*.md` | `drop_bad`, `empty_char_threshold`, `empty_min_pages` (`threshold` is deprecated, no effect) | `clean_markdown/<stem>.md`, cleaner report parquet, parquet flags such as `filter` and `needs_ocr` |
| OCR retry | `Corpus.ocr(mode='ocr_bad'...)` | pages page quality routes (default `page_routing="all"`), or only cleaner-flagged documents with `page_routing="flagged"` | `mode`, `fix_bad`, `page_routing`, `use_gpus`, `devices` | refreshed `markdown/<stem>.md`, refreshed cleaner/parquet metadata |
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
  - `repair_fonts=True` (default): worker-local CPU ToUnicode repair before safe or Docling extraction; `False` uses the original text layer
  - `use_gpus='single'|'multi'`
  - `workers_per_device` to fan out more than one extraction worker onto each GPU
  - `export_doc_json` and `emit_formula_index` for later Phase‑2 work
- Operational note:
  - `force_ocr` is deprecated and ignored in Phase‑1; use `Corpus.ocr(ocr_options=plugin_options)` after `clean()` for OCR remediation
- Main outputs:
  - canonical markdown in `markdown/<stem>.md`
  - repaired copies in `repaired_pdfs/<stem>.pdf`; originals in `downloads/` are preserved
  - parquet `font_repair_status`, `font_repair_fonts`, `font_repair_valid_rate_before` and `font_repair_valid_rate_after` (rates use first 20 pages when `GLOSSAPI_GREEK_WORDLIST` is configured; otherwise null)
  - every extraction derives the repair from the source again (an existing `repaired_pdfs/<stem>.pdf` is never reused by modification time); pre-pass errors/timeouts fall back to the original, using the shared file budget (see [Extract stage](stages/extract.md))
  - `problematic_files/` and `timeout_files/` (debug) are created by extraction for files it could not convert
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
  - `drop_bad`: remove the documents the fixed gates reject (`mojibake_badness_score` above 0.1, or Greek badness above 60) from downstream selection
  - `empty_char_threshold` and `empty_min_pages` for OCR fallback decisions
  - `threshold` is deprecated and has no effect; passing it raises a `DeprecationWarning`
- Main outputs:
  - cleaned markdown in `clean_markdown/`
  - updated parquet metadata with quality and OCR-related flags
- Resume state:
  - interrupted streaming runs resume from the ledger (`glossapi run --resume RUN_ID`); the metadata parquet is the compatibility state the phase methods read and write. No pickle file is written or read.

### 4. OCR Retry and Phase‑2 Enrichment

- Main code: `Corpus.ocr()` and `Corpus.formula_enrich_from_json()`
- Purpose:
  - OCR the pages page quality routes (`page_routing="all"`, the default, as `glossapi run` does), or only the documents the cleaner flagged (`page_routing="flagged"`)
  - optionally decode formula/code regions from Docling JSON into markdown
- Modes:
  - `ocr_bad`
  - `math_only`
  - `ocr_bad_then_math`
- Main outputs:
  - refreshed `markdown/<stem>.md`
  - `json/<stem>.latex_map.jsonl` when math/code enrichment runs

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
- debug — extra files kept around when something fails or needs a closer look
- ledger — the provenance ledger and the private work area of `Corpus.run()` / `glossapi run`

OUT/
├── downloads/                                  (canonical)
│   └── problematic_math/                       (debug)
├── repaired_pdfs/                              (sidecars: repaired extraction inputs)
│   └── <stem>.pdf
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
├── sidecars/extract/<stem>.json                (per-document extraction metadata)
├── sections/                                   (canonical)
│   └── sections_for_annotation.parquet
├── classified_sections.parquet                 (canonical)
├── fully_annotated_sections.parquet            (canonical)
├── export.jsonl                                (canonical, written by the export stage of a run)
├── corpus.sqlite3                              (ledger)
├── .glossapi/artifacts/                        (ledger: private stage outputs of a run)
├── problematic_files/                          (debug, extraction)
└── timeout_files/                              (debug, extraction)

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
