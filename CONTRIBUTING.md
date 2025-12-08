# Contributing to GlossAPI

## Working branches and PR flow
- Open PRs against the `development` branch. Create your feature/fix branch from `development`, push it, and target `development` in the PR so release branches (`master`/`main`) stay clean.
- If an urgent hotfix lands on `master`/`main`, mirror it back into `development` right away to keep the branches aligned.

## Pipeline awareness and folder layout
- Tie any pipeline change to the artifacts it produces. Common touchpoints:
  - `Corpus.extract()` writes source PDFs under `downloads/` and a manifest at `download_results/download_results.parquet` (fields like `needs_ocr`).
  - `Corpus.clean()` emits `markdown/` and `clean_markdown/`, keeping `.processing_state.pkl` plus `problematic_files/` and `timeout_files/` subfolders.
  - `Corpus.ocr()` and `Corpus.section()` populate `json/` (Docling JSON, formula index, metrics) and `sections/sections_for_annotation.parquet`.
- When relocating outputs or adding new ones, update assertions in `tests/test_pipeline_smoke.py` and the folder references in `docs/pipeline.md` so the layout stays discoverable.

## Keep changes small
- Avoid large refactors or sweeping interface changes; aim for narrowly scoped PRs and discuss big shifts before starting.
