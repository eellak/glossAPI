# OCR Noise Failure Modes

Status: example bank for future `Corpus.clean_ocr(...)` heuristics. These are notes only, not implemented cleaning rules.

## Why This Exists

The preserved OCR outputs contain several distinct failure modes that should not be collapsed into one generic `ocr_noise` rule. Some are page-local low-entropy collapses, some are encoding/control-character tails, and some are repetitive math-token artifacts that need math-aware handling.

The examples below were reviewed on April 3, 2026 from the preserved OCR lane:

- `/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown`

## Group 1: Page-Local Low-Entropy Numeric Collapse

Definition:
pages that collapse into highly repetitive short numeric lines, often immediately after a page split marker.

Examples:

- `ABO_768__p00001-00096.md`
  - around line 955 the page turns into repeated `0`, `0 0`, `0 0 0`
  - the collapse begins directly after `<--- Page Split --->`
- `ACH_787__p00001-00096.md`
  - around line 755 the page turns into repeated `1.1` and occasional `1`
  - this also begins directly after `<--- Page Split --->`

Anchored references:

- [ABO_768__p00001-00096.md:955](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ABO_768__p00001-00096.md#L955)
- [ACH_787__p00001-00096.md:755](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ACH_787__p00001-00096.md#L755)

Detection ideas:

- page-level repeated-line detection, not just single-line run detection
- low token entropy on a page-sized region
- special weight if the collapse starts right after `<--- Page Split --->`
- repeated short numeric lines should be treated separately from legitimate tables or lists

Important note:
the current OCR numeric-noise check is line-local and is better at catching long same-number or ascending sequences inside one line than these repeated-line page collapses.

## Group 2: Control-Character / Encoding-Garbage Tails

Definition:
pages that devolve into non-printable or control-like characters, often after otherwise valid text.

Example:

- `ADQ_670.md`
  - after a page split, the page contains `%` followed by C1/control-like junk such as ``, ``, ``, ..., `°`
  - this is not just numeric repetition; it looks like decoding/binary leakage or severe mojibake-like corruption

Anchored references:

- [ADQ_670.md:887](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ADQ_670.md#L887)
- [ADQ_670.md:954](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ADQ_670.md#L954)

Detection ideas:

- count non-printable/control codepoints
- count dense runs of extended control-like characters on a page
- flag abrupt transitions from valid prose to control-character tails
- keep this separate from ordinary mojibake and separate from numeric collapse

## Group 3: Repetitive Math-Token Floods

Definition:
pages or page segments that repeat the same LaTeX-like math atoms or malformed math atoms many times.

Examples:

- `ADS_856__p00001-00014.md`
  - repeated `\( \gamma \)` sequence on one line
- `ADS_856__p00015-00082.md`
  - repeated `\( \Delta_{v} \)` blocks
  - malformed variants like `\( \Deltav \)`
  - long concatenated runs like `\Delta_{v}\Delta_{v}\Delta_{v}...`

Anchored references:

- [ADS_856__p00001-00014.md:139](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ADS_856__p00001-00014.md#L139)
- [ADS_856__p00015-00082.md:1](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ADS_856__p00015-00082.md#L1)

Detection ideas:

- tokenize LaTeX-like math atoms and detect repeated-token floods
- distinguish valid repeated notation from pathological repetition
- score malformed math variants separately from valid math tokens
- this should remain an experimental detector, not a blunt drop rule

Important note:
real mathematical texts can legitimately repeat symbols, so this class needs a math-aware heuristic rather than a general repetition penalty.

## Grouping Recommendation

Do not collapse all of the above into one rule.

Recommended future flags:

- `ocr_numeric_page_collapse`
- `ocr_control_char_tail`
- `ocr_math_repetition`

Recommended future metadata:

- page-local region counts
- page-split proximity flags
- repeated-line entropy or uniqueness ratio
- control-character density
- math-token repetition density

## Current Examples To Keep Around

- [ABO_768__p00001-00096.md:955](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ABO_768__p00001-00096.md#L955)
- [ACH_787__p00001-00096.md:755](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ACH_787__p00001-00096.md#L755)
- [ADQ_670.md:887](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ADQ_670.md#L887)
- [ADS_856__p00001-00014.md:139](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ADS_856__p00001-00014.md#L139)
- [ADS_856__p00015-00082.md:1](/home/foivos/data/openarchives_ocr_ingest_20260403/lanes/eu_node01_full_v1/markdown/ADS_856__p00015-00082.md#L1)
