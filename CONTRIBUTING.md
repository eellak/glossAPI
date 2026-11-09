# Contributing to GlossAPI

## Working branches and PR flow

Open focused pull requests against `master`, the release branch. Create short-lived feature/fix branches from
`master`; the archived dependency campaign is separate from the 0.2.0 release. Keep changes small and discuss
large refactors or public interface changes before implementation.

## Design principles

- Keep pipeline changes accessible through descriptive `Corpus` methods and the `glossapi` CLI.
- Keep Python files readable and organized by responsibility.
- Preserve user metadata and the typed metadata policy. Runtime diagnostics and provenance belong in the corpus
  ledger and the documented sidecars; avoid adding hidden state or separate resume authorities.
- The supported OCR implementation is the pinned Greek OCR toolkit. Core layout-aware Markdown does not require
  Docling; optional Docling layout/math uses the `layout` extra. Downloads accept public targets only.

## Pipeline awareness and folder layout

Tie every pipeline change to its artifacts and tests:

- `Corpus.download()` acquires documents into `downloads/` and updates `download_results/download_results.parquet`.
- `Corpus.extract()` writes `markdown/` and extraction/page sidecars. The cleaner writes `clean_markdown/` and
  metrics/flags; processing-state persistence uses typed metadata, not the old pickle checkpoint.
- `Corpus.ocr()` replaces selected physical pages or records a supplement fallback. Docling JSON/formula artifacts
  require optional Docling extraction. `Corpus.section()` writes `sections/sections_for_annotation.parquet`.
- `Corpus.run()` / `glossapi run` use `corpus.sqlite3` for frozen plans, attempts, hashes, recovery and publication.
  See [resumability and retention](docs/architecture/resumability_recovery_and_retention.md).

Update [pipeline contracts](docs/pipeline.md), [API reference](docs/api/corpus.md), [code map](docs/code_map.md) and
relevant regression assertions when changing outputs or signatures. Start with
`pytest tests/test_corpus_phases_lightweight.py`; run the affected tests and the staged typing gate
(`python tools/typing/check_typing.py`). Build distributions with `python -m build` and exercise archive regressions
with `GLOSSAPI_SDIST=dist/glossapi-0.2.0.tar.gz GLOSSAPI_WHEEL=<built-wheel> pytest tests/test_sdist_contract.py`.
Real OCR tests require explicit integration flags and model caches. Keep private documents, credentials and experiment
outputs out of source control; a skipped optional integration test is not evidence that the integration passed.
