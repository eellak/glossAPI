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

## Extraction failures and page routing

Every metadata row retains `extract_status` from `sidecars/extract/<stem>.json`.
Legacy sidecars without a status use `failure_mode`, then the existing parquet
status; absent provenance is explicitly `unknown`. Extraction workers publish
status and source `page_count` for PDFium and Docling, and the controller merges
these into parquet. Safe PDFium extraction stays on PDFium above 600 pages.

For PDFs, a `problematic`, `timeout`, `failed`, `failure`, or `error` extraction,
or missing input Markdown, adds the `extract_failed` filter reason and sets
`needs_ocr=True`, even if quality metrics are unknown or an old `ocr_success`
flag is present. These rows retain an empty `ocr_pages` route and a zero
`ocr_pages_count`: OCR uses the whole-document fallback. An empty Markdown file
that exists instead goes through the existing `is_empty` gate.

Valid page-quality sidecars attach `ocr_pages` and `ocr_pages_count` to both
flagged and unflagged PDFs without promoting the document-level `needs_ocr`
flag. `ocr_pages_unflagged_count` records selected pages inside unflagged
documents in both routing modes. `Corpus.ocr(page_routing="flagged")` is the
default and dispatches only flagged documents; `page_routing="all"` additionally
dispatches the non-empty routes of unflagged documents. A flagged document
whose valid page-quality report routes no page is not dispatched (as `glossapi
run`, K5 quality policy P4d); a flagged document without a report keeps the
whole-document fallback. Existing completion and skip-list gates
continue to apply. Explicit reruns of completed, unflagged documents retain
whole-document behavior in `flagged` mode; only `all` applies their retained
page routes. These counts describe selection, not successful execution.

`pua_char_count` counts BMP private-use characters U+E000–U+F8FF in the input
Markdown. `control_char_count` counts C0/C1 controls U+0000–U+001F and
U+007F–U+009F, excluding normal Markdown whitespace TAB, LF and CR. Both metrics
are nullable when Markdown is absent. Page-quality rows expose the same inputs;
PUA/control dominance above the symbol-garbage threshold routes sparse corrupt
text before the blank-page gate. Counts alone do not establish that every PUA
character is erroneous, and the cleaner does not flag an entire document merely
because it contains PUA characters.

Text metrics are measured on the markup-free text (K5-A2b, K5A2A-G2): for a
layout-labelled Markdown (the safe backend's default) the extract sidecar's
`markdown_layout.strip` record gives the plain text back
(`glossapi.markdown_layout.markup_free`), and the native cleaner report
(`mojibake_badness_score`, `percentage_greek`, `latin_percentage`,
`char_count_no_comments`, `is_empty`), the noise scores (`greek_badness_score`,
`polytonic_ratio`) and the raw-text measurements (`latin_word_shape_rate`,
`pua_char_count`, `control_char_count`, the glyph check below) are taken from
it. The cleaned Markdown is still cleaned from the labelled Markdown and keeps
its markup. A record that does not describe the file (changed text, another
file) is not applied. Docling Markdown has no record and is measured as before.
After OCR, `clean_ocr` measures its OCR profile on the markup-free text and
also cleans that text; the edits from its cleaned (labelled) output to that
cleaned plain text are kept as a strip record of the cleaned file (a recovery
artifact of the clean attempt, or in memory on the legacy profile), and the
score-only `clean` that follows measures the cleaned plain text through it. A
score-only pass in a new process without that record falls back to the extract
sidecar's record, which describes the cleaned file only when the OCR cleaner
left it unchanged.

All-GLYPH text (K5 quality policy P4a): a document whose text is entirely glyph
names or unusable glyph codes has no readable characters and is routed to OCR
with the filter token `glyph_only` (`needs_ocr=True`, reason code
`clean_glyph_only`). "Entirely" means at least 10 glyph-name matches covering at
least 90% of the text's non-whitespace characters, measured with the cleaner's
own Rule A/B matchers on the raw (markup-free) text
(`glossapi_rs_cleaner.glyph_residue_counts`, native API level 3). The cleaner
itself still drops such lines before it scores, so the mojibake score of the
cleaned text stays low.
