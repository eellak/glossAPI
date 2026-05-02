# OCR Cleaning Runtime

This document explains how the current OCR cleaner is organized, why the
matcher families are separated, and why the clean/debug behavior is driven by
one shared page analyzer.

## One Analyzer, Two Render Modes

The OCR cleaner now works in two modes over the same span plan:

- `debug`
  - preserves the source page surface
  - inserts `<match ...>` tags around the matched regions
- `clean`
  - applies the removal/rewrite policy directly
  - writes the cleaned page text with no debug tags

This is deliberate. The project previously had a tendency for the reviewer-facing
debug logic to evolve faster than the real cleaner. Sharing one analyzer avoids
that drift: if the debug page is right, the clean page is operating on the same
decisions.

The same alignment rule applies to metadata. When `clean_ocr()` writes cleaned
markdown, it now scores parquet-facing OCR metrics from that cleaned directory.
Without that, downstream `export` can read cleaned text while still carrying
raw-OCR `char_count_no_comments`, `is_empty`, or repeat diagnostics in parquet.

## Code Layout

The OCR cleaner is split by responsibility so detector policy does not drift
into rendering or table-specific cleanup:

- `src/glossapi/corpus/phase_clean.py`
  - owns analyzer order, worker orchestration, and clean/debug mode selection
- `src/glossapi/corpus/ocr_table.py`
  - owns HTML-table classification, dropping policy, and HTML->Markdown conversion
- `src/glossapi/corpus/ocr_render.py`
  - owns span merging, clean/debug rendering, and `match_index.jsonl` row generation
- `src/glossapi/corpus/text_surface_metrics.py`
  - owns shared published-surface metric helpers such as `char_count_no_comments`
    and `is_empty`

This split is deliberate. The project needs one place to decide *what* a match
is, and a separate place to decide *how* that decision becomes visible output.
That boundary is one of the main protections against random debug/clean drift.

## Stage Boundary: `clean_ocr()` vs `clean()`

These two stages deliberately remain separate:

- `clean_ocr()`
  - sits on the `corpus.ocr` path
  - owns OCR artifact removal and OCR-specific metrics
- `clean()`
  - sits primarily on the `corpus.extract` path
  - owns broader clean/export quality metrics

The OCR rerun path reuses `clean()` afterward because `export` still expects the
generic clean/export fields. That reuse is intentional orchestration, not an
argument that the two stages should collapse into one function.

## Field Ownership

The practical parquet contract is:

- OCR-owned fields
  - `percentage_greek`
  - `latin_percentage`
  - `polytonic_ratio`
  - `char_count_no_comments`
  - `is_empty`
  - `ocr_noise_suspect`
  - `ocr_noise_flags`
  - `ocr_repeat_phrase_run_max`
  - `ocr_repeat_line_run_max`
  - `ocr_repeat_suspicious_line_count`
  - `ocr_repeat_suspicious_line_ratio`
- clean/export-owned fields
  - `greek_badness_score`
  - `mojibake_badness_score`
  - `needs_ocr`
  - `filter`
  - other generic quality-routing fields

This ownership split is the reason the post-OCR sequence is explicit:

1. `clean_ocr()` updates the OCR-owned layer on the OCR-cleaned text surface.
2. `clean(..., write_cleaned_files=False)` refreshes the generic clean/export
   layer on that same surface.

## Why The Cleaner Is Not One Generic Matcher

The cleaner is trying to remove OCR- or VLM-induced garbage, not every repeated
pattern in a page. A single fuzzy matcher over the whole page overgeneralizes
quickly:

- numbers steal matches that should belong to numeric progression logic
- repeated notation in LaTeX looks like corruption even when it is legitimate
- HTML tables distort text surfaces and cause spurious word matches

So the runtime uses ownership by surface type and structure instead of one broad
"repetition" rule.

## Page Ownership Order

The current analyzer order is:

1. tables
2. numeric
3. LaTeX
4. hybrid numbered repetition
5. shared text repetition

Why this order:

- Tables run first because HTML table shells can dominate a page and confuse
  every later pass.
- Numeric runs before generic text because `1, 2, 3, ...` style progressions
  are real OCR-collapse signals and should not be absorbed by `word_repeat`.
- LaTeX and hybrid passes run before generic text because they depend on local
  structure, not just repeated tokens.
- Shared text repetition runs last on the remaining visible surface only.

This ordering is the main false-positive control mechanism.

## Table Cleaning Is Broader Than Repetition

Table handling is intentionally separated into `src/glossapi/corpus/ocr_table.py`
because it is not just another repetition matcher.

Current table classes:

- `sentence_shell_table`
  - a table with one prose-like filled cell
  - treated as layout noise around content
  - dropped in clean mode
- `empty_table_collapse`
  - a large sparse shell with almost no real cell content
  - dropped in clean mode
- `repeated_rows`
  - an actually repetition-oriented table problem
  - dropped in clean mode
- unmatched kept tables
  - converted from HTML to GitHub-style Markdown

The important design point is that sentence-shell and empty-shell tables are
structural cleanup decisions, not repetition decisions.

## LaTeX And Hybrid Generalization Strategy

LaTeX and hybrid numbered matching both follow the same conservative pattern:

- prefer local runs
- abstract slot fields
- require mechanical progression or stable low-diversity cycles
- avoid page-wide reuse as evidence on its own

That is why the cleaner does not treat "same symbol appears many times on a
page" as enough evidence. The goal is to catch degenerate local collapse, not
normal scholarly notation reuse.

## Why Rust Is Used Selectively

The hot-path detection work is in Rust because page-scale scanning dominates run
time. Python still owns:

- orchestration
- filesystem I/O
- debug/clean rendering
- policy composition across matcher families

This split is intentional:

- Rust is best for large repeated scans and token-normalization hot loops
- Python is still easier for mode-aware rendering and pipeline integration

## Performance And Correctness Contract

Performance work is allowed only if exact debug output stays stable.

The correctness lock is:

- `tests/test_ocr_golden_pages.py`

That suite uses hundreds of real pages and compares exact output bytes. The
speed work therefore optimizes implementation, not semantics.
