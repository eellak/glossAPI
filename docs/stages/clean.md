# Clean Stage

## Purpose

The clean stage normalizes extracted Markdown and evaluates its quality.

## Main responsibilities

- run the shared OCR analyzer in either clean or debug rendering mode
- compute text quality and badness metrics
- detect documents that require OCR reruns
- update metadata for downstream stage selection

## Main inputs

- Markdown in `markdown/`
- metadata parquet
- cleaner configuration and thresholds

## Main outputs

- cleaned Markdown in `clean_markdown/`
- debug-marked Markdown under `debug/` when debug output is requested
- debug manifests under `debug/`:
  - `manifest.jsonl`
  - `page_metrics.jsonl`
  - `match_index.jsonl`
  - `summary.json`
- quality metrics and reports
- metadata updates including OCR-related decisions

## Why this stage is critical

This stage is the main bridge between extraction and corrective OCR.

It is especially important for Greek corpora because it distinguishes:

- technically extracted text
- actually usable Greek text

It also separates two different responsibilities that are easy to conflate:

- structural cleanup
  - tables, numeric runs, LaTeX collapse, hybrid numbered loops, word repetition
- quality scoring
  - bad-character metrics
  - suspicious-line metrics
  - OCR rerun recommendations

The stage now uses one shared analyzer for both:

- `debug` mode
  - shows exact match placement with `<match ...>` tags
  - records merged-span match metadata in `match_index.jsonl`
- `clean` mode
  - removes or rewrites those exact same matched regions
- `clean + debug`
  - writes pipeline-ready cleaned Markdown and the parallel debug artifacts from the same span plan in one run

When `clean_ocr()` writes cleaned Markdown, parquet refresh is now scored from
that cleaned directory, not from the raw OCR markdown. This is deliberate:
`export` prefers `clean_markdown/`, so `char_count_no_comments`, `is_empty`,
`percentage_greek`, and the OCR-repeat diagnostics need to describe the same
surface that downstream stages publish.

## Important operational outputs

This stage may contribute or update:

- quality scores
- filter classifications
- `needs_ocr`
- character-count-based diagnostics
- processing-stage status

For OCR reruns, the intended pipeline contract is now:

1. `clean_ocr()` writes OCR-cleaned markdown and OCR-specific repetition/noise metrics
2. `clean(..., write_cleaned_files=False)` refreshes generic clean/export metrics
   such as `greek_badness_score` against that OCR-cleaned text

This keeps `corpus.ocr`, `clean_ocr`, and `export` aligned on one publish
surface instead of mixing cleaned text with stale parquet values.

## Main code layout

The clean stage now has an explicit internal split:

- `src/glossapi/corpus/phase_clean.py`
  - analyzer order, orchestration, worker setup, and stage entrypoints
- `src/glossapi/corpus/ocr_table.py`
  - table-specific structural cleanup and Markdown conversion
- `src/glossapi/corpus/ocr_render.py`
  - merged-span rendering for both `clean` and `debug`, plus match indexing
- `src/glossapi/corpus/text_surface_metrics.py`
  - shared published-surface helpers such as `char_count_no_comments` / `is_empty`

This is a maintainability boundary, not a semantic one. The analyzer still owns
all cleaning decisions; rendering modules only turn those decisions into page
text and debug artifacts.

## Stage ownership

The names are similar, but the stages are not synonyms:

- `clean_ocr()`
  - intended for `corpus.ocr`
  - removes OCR-specific artifacts
  - updates OCR-owned metrics on the OCR-cleaned text surface
- `clean()`
  - intended mainly for `corpus.extract`
  - computes broader text-quality and export-facing metrics
- `clean()` after `clean_ocr()`
  - deliberate reuse, not stage collapse
  - refreshes export-facing metrics against already OCR-cleaned text

## Metadata ownership

The parquet update contract is intentionally split by stage ownership:

| Owner | Fields / responsibility |
| --- | --- |
| `clean_ocr()` | `percentage_greek`, `latin_percentage`, `polytonic_ratio`, `char_count_no_comments`, `is_empty`, `ocr_noise_suspect`, `ocr_noise_flags`, `ocr_repeat_phrase_run_max`, `ocr_repeat_line_run_max`, `ocr_repeat_suspicious_line_count`, `ocr_repeat_suspicious_line_ratio` |
| `clean()` | general clean/export metrics such as `greek_badness_score`, `mojibake_badness_score`, `needs_ocr`, `filter`, and related quality routing fields |
| `export` | reads the latest parquet values plus `clean_markdown/` when present; it does not own OCR-cleaning decisions |

This ownership split is why the OCR rerun path now does:

1. `clean_ocr()`
2. `clean(..., write_cleaned_files=False)`

The first call refreshes OCR-specific cleanup and OCR metrics. The second call
refreshes the normal clean/export layer on top of that OCR-cleaned surface.

## Current cleaning policy

The cleaner does not use one generic fuzzy matcher over the whole page.
Instead it applies ownership in a fixed order:

1. tables
2. numeric
3. LaTeX
4. hybrid numbered repetition
5. shared word repetition

Why this matters:

- tables can distort the visible text surface for every later pass
- numeric progressions are often valid cleaner targets but should not be
  consumed by generic text repetition
- LaTeX and hybrid passes rely on more specific local structure
- shared text repetition is therefore safest on the remaining surface only

Table handling is intentionally broader than repetition:

- `sentence_shell_table` is dropped
- `empty_table_collapse` is dropped
- `repeated_rows` is dropped
- unmatched tables are converted from HTML to GitHub-style Markdown

## Failure concerns

Typical issues include:

- missing Rust extension
- metadata bootstrap issues
- misleadingly non-empty but low-quality Markdown

## Contributor note

Changes here affect OCR routing and post-run quality analysis. Treat score and flag semantics as contract-level behavior.

For content-cleaning changes, the exact-output benchmark in
`tests/test_ocr_golden_pages.py` is the main regression lock. Speed work is only
acceptable if those outputs remain stable.

For stage-level speed and byte-level parity checks on raw OCR markdown, use:

- `scripts/bench_ocr_stage_subset.py`

That script benchmarks one stage at a time (`clean`, `clean_ocr`,
`clean_ocr_debug`) against a fixed file manifest or a bounded raw-markdown
subset, and writes hashes plus parquet rows to `summary.json`.
