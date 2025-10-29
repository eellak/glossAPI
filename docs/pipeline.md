# Pipeline Overview & Artifacts

GlossAPI is a staged pipeline. You can enter at any stage and use the same folder for input and output.

## Stages

- Download (optional): fetch source files from URLs → `downloads/`
- Extract (Phase‑1): parse PDFs to Markdown; optional GPU OCR → `markdown/<stem>.md`
- Clean: compute quality metrics and filter low‑quality items; decide which to OCR
- OCR (compat shim): re‑run extract on filtered items with `force_ocr=True`
- JSON + index (optional): emit `json/<stem>.docling.json(.zst)` and `json/<stem>.formula_index.jsonl` for Phase‑2
- Enrich (Phase‑2): decode FORMULA/CODE from JSON on GPU → overwrite `markdown/<stem>.md`, write `json/<stem>.latex_map.jsonl`
- Section: produce `sections/sections_for_annotation.parquet`
- Annotate: classify sections; produce `classified_sections.parquet` and `fully_annotated_sections.parquet`

## Artifact Layout

```
OUT/
├── downloads/
│   └── problematic_math/
├── download_results/
├── markdown/
│   └── <stem>.md
├── json/
│   ├── <stem>.docling.json(.zst)
│   ├── <stem>.formula_index.jsonl
│   ├── <stem>.latex_map.jsonl
│   ├── metrics/
│       ├── <stem>.metrics.json
│       └── <stem>.per_page.metrics.json
│   └── problematic_math/
├── sections/
│   └── sections_for_annotation.parquet
├── classified_sections.parquet
└── fully_annotated_sections.parquet
```

Notes:
- Enriched Markdown replaces the plain Markdown (single canonical location).
- Metrics lived under `markdown/` in earlier versions; they now live under `json/metrics/`.
- When math enrichment cannot recover after the configured number of respawns, the corresponding PDFs and Docling artifacts are copied into the `problematic_math/` folders above and the stems are added to the fatal skip-list for later review.

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
