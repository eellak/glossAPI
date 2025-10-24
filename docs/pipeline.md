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
