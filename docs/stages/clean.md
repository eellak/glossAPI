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
- debug-marked Markdown when using the debug helpers
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
- `clean` mode
  - removes or rewrites those exact same matched regions

## Important operational outputs

This stage may contribute or update:

- quality scores
- filter classifications
- `needs_ocr`
- character-count-based diagnostics
- processing-stage status

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
