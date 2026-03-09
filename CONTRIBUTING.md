# Contributing to GlossAPI

## Working branches and PR flow
- Open PRs are pushed against the `development` branch.
- Development is merged with master when a) everything has been effectively used a few times and b) we reach a clear checkpoint.  

## Some design principles
- Corpus methods should be easy to use and descriptive.
- Python files should be readable and well organized (check folder structure).
- Metadata should be written to two distinct parquet files depending on their relevance to the end user ("metadata") or debugging during pipeline runs. The principle of reading/ writing to  these parquet files should be maintained through out. Rest of the metadata is implicitly encoded in the output folders at each stage of the pipeline.

## Pipeline awareness and folder layout
- Tie any pipeline change to the artifacts it produces. Common touchpoints:
  - `Corpus.extract()` writes source PDFs under `downloads/` and a manifest at `download_results/download_results.parquet` (fields like `needs_ocr`).
  - `Corpus.clean()` emits `markdown/` and `clean_markdown/`, keeping `.processing_state.pkl` plus `problematic_files/` and `timeout_files/` subfolders.
  - `Corpus.ocr()` and `Corpus.section()` populate `json/` (Docling JSON, formula index, metrics) and `sections/sections_for_annotation.parquet`.
- When relocating outputs or adding new ones, update assertions in `tests/test_pipeline_smoke.py` and the folder references in `docs/pipeline.md` so the layout stays discoverable.

## Keep changes small
- Avoid large refactors or sweeping interface changes; aim for narrowly scoped PRs and discuss big shifts before starting.
