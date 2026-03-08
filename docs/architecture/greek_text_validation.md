# Greek Text Validation

Greek text validation is a core quality gate in GlossAPI.

This is one of the most important parts of the pipeline because many Greek PDFs fail in ways that are not captured by simple extraction success.

## Why this exists

Many Greek PDFs have one or more of the following problems:

- no `ToUnicode` mapping
- broken `ToUnicode` mapping
- incorrect text-layer encoding
- text that is technically extractable but not semantically valid Greek

This means a document can produce Markdown that looks operationally successful while still being unusable.

## Why plain extraction success is insufficient

A successful file write does not prove:

- the output is readable Greek
- the output represents the original document correctly
- the text layer is reliable enough to avoid OCR

This is why the pipeline separates:

- extraction
- quality validation
- OCR remediation

## What the cleaner contributes

The cleaning stage uses Rust-backed metrics and heuristics to evaluate extracted text quality.

The cleaner does more than normalize Markdown. It also helps answer:

- does this output look like valid Greek text?
- is the output mostly Latin or mojibake?
- is the document effectively empty after cleanup?
- should this file be rerouted through OCR?

## Important metadata outcomes

Several metadata outputs are operationally important for Greek validation:

- badness or quality scores
- `needs_ocr`
- filter classification
- character-count-related diagnostics
- OCR success after remediation

These fields are not just reporting detail. They determine downstream behavior.

## Why OCR is selective

For Greek corpora, OCR is necessary for many files but still should not be the default for every file.

Selective OCR is important because:

- text-layer extraction is cheaper
- many files do not need OCR
- OCR introduces additional runtime and failure surface

So the current strategy is:

1. extract normally
2. validate text quality
3. OCR only the files that look invalid or unreliable

## Common failure patterns in Greek corpora

Operators and contributors should expect issues such as:

- consonant-heavy garbage output
- mostly Latin text where Greek should exist
- low-information or almost-empty Markdown
- incorrect sigma or bigram patterns
- severe mojibake

These are not niche anomalies. They are part of the expected workload.

## Why this should be documented prominently

Without this context, a new contributor may assume:

- extraction already solved the problem
- OCR is optional polish
- quality validation is just cleanup

All three assumptions would be wrong for the target corpus.

Greek validation is one of the main reasons the pipeline exists in this form.

## Operator guidance

When evaluating a run, operators should be able to answer:

- which files passed extraction but failed Greek quality checks
- which files were marked `needs_ocr`
- which files improved after OCR
- which files remain problematic even after OCR

## Contributor guidance

Any change to extraction, cleaning, OCR routing, or scoring should be evaluated against this question:

- does this make the pipeline better or worse at distinguishing valid Greek from technically successful but unusable output?

If that question is not addressed, the change is incomplete.
