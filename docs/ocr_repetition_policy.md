# OCR Repetition Policy

This document pins the intended default repetition thresholds for OCR-cleaner development so they do not drift silently.

## Defaults

- Shared word repetition threshold: `4`
- Shared LaTeX repetition threshold: `4`
- Shared minimum repeat period: `3`
- Shared repeat window: `96`

These defaults apply to the combined OCR debug annotator:
- `Corpus.clean_ocr_numeric_word_debug_docs(...)`

In that pipeline:
- numeric detection runs first
- LaTeX detection runs second
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

- A default of `4` is meant to reduce borderline `3`-repeat matches.
- Locality matters more than page-wide reuse, especially for LaTeX.
- Repeated symbols or notation used normally across a page should not be treated as cleaner targets by default.
