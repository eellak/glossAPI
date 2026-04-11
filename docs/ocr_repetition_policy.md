# OCR Repetition Policy

This document pins the intended default repetition thresholds for OCR-cleaner development so they do not drift silently.

## Defaults

- Shared word repetition threshold: `4`
- Shared LaTeX repetition threshold: `4`
- Shared minimum repeat period: `3`
- Shared repeat window: `96`

These defaults apply to the combined OCR debug annotator:
- `Corpus.clean_ocr_numeric_word_debug_docs(...)`

The same analyzer now also drives real clean-mode rendering in `clean_ocr()`;
debug and clean differ only in rendering, not in span discovery.

In that pipeline:
- tables are handled first
- numeric detection runs before generic text ownership
- LaTeX and hybrid structural detection run before shared text repetition
- shared repeat detection runs last on the remaining untagged text

## Scope

These defaults are for:
- word repetition
- LaTeX repetition

They do not override numeric-specific detectors, which have their own thresholds such as:
- ascending numeric progressions
- compact repeated numeric atoms
- same-digit numeric runs

## Design Intent

- Neighboring same-type spans may merge when their separator has `40` non-whitespace characters or less; this keeps fragmented OCR loops from being split into multiple tiny matches.
- A default of `4` is meant to reduce borderline `3`-repeat matches.
- Locality matters more than page-wide reuse, especially for LaTeX.
- Repeated symbols or notation used normally across a page should not be treated as cleaner targets by default.
- Numeric progression should be handled by numeric or hybrid logic before text repetition sees it.
- Table cleanup includes structural cases that are not repetition problems, so table policy is documented separately in `docs/architecture/ocr_cleaning_runtime.md`.
