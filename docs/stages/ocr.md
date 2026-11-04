# OCR remediation

Run extraction and cleaning first, then call
`corpus.ocr(mode="ocr_bad")`. The default is the pinned external Greek R8+x1
plugin. Install it with `python install_glossapi.py ocr --device cpu` (or
`mps` / `cuda126` / `cuda132`). Explicit `ocr_options` still select a configured
plugin; there is no silent engine fallback. `glossapi run --stages ocr,export`
also supports this default and routes only pages flagged by page quality.

The toolkit registers `glossapi = "greekocr.deploy.docling_plugin.docling_plugin"`
in the Docling `docling` entry-point group. Docling loads `ocr_engines()` with
`allow_external_plugins=True`; the engine owns its options, model files and inference.

`page_routing="all"` (the default for the OCR modes) uses the page policy of
`glossapi run`: every page that page quality routes is OCRed, in documents the
cleaner flagged and in documents it did not. `page_routing="flagged"` restricts
OCR to the documents the cleaner flagged (`needs_ocr`), the default before K4g.
`mode="math_only"` performs no OCR and by default still skips only
cleaner-flagged documents. Ranges are inclusive, one-based physical pages. A
cleaner-flagged document whose page quality routed no page is not OCRed by
`Corpus.ocr` in either setting, as `glossapi run` keeps its extracted text
(`ocr_not_routed`, cause `flagged_without_routed_pages`; K5 quality policy P4d).
A document without a page-quality report keeps document-level routing: the OCR
stage's own page quality picks its pages. A document whose native text layer is
entirely glyph names (the cleaner's `glyph_only` token, K5 quality policy P4a)
is OCRed whole, in `Corpus.ocr` and in `glossapi run` alike: page quality v4
does not route such non-Greek symbol pages, so the OCR stage's glyph check
selects every page (decision `route: glyph_only`). Skip-list and completed-document filtering still apply.
Partial OCR replaces selected physical pages in place, without adding page markers.
The default extractor's `markdown_layout.pages` offsets are checked against its
strip record's full Markdown hash; plain extraction records equivalent hash-bound
`plain_pages` offsets. Blank pages keep their physical page numbers. Missing,
stale or unusable offsets retain the original and append the existing supplement;
`page_replacement_fallback_reason` in the OCR metrics explains the fallback.
Legacy explicit page boundaries and checked Docling provenance remain supported.
The latest replacement per physical page is retained on reruns. The immutable
native snapshot and committed OCR page state allow a killed attempt to restart.

Replacement writes `sidecars/ocr_layout/<stem>.json`: a new hash-bound strip of
the assembled Markdown. Only retained native markup is stripped for metrics;
OCR text remains the engine's text. Both the run DAG and compatibility cleaner
bind this provenance, including the post-OCR cleanup and scoring passes.

OCR attempts use an absolute deadline of `max(600, min(7200, 120 + 45*N))`
seconds, where N is the routed page count (bounded by `max_pages`). The scheduler
computes it before dispatch and records the policy in `ocr_deadline` attempt
settings. A supplied `ocr.timeout_seconds` overrides it. The deadline remains
fixed during an attempt and the worker watchdog terminates hangs; an active
attempt cannot renew its budget.

Routing v4 remains the default. The existing `rule_profile` switch also accepts
experimental `v5` (`settings.page_quality.rule_profile` in `run`, or
`page_quality(..., rule_profile="v5")`). It preserves v4 and adds measured
English validity on Latin-dominant text layers with at least 50 characters:
less than 40% recognized Latin tokens selects OCR. Every effective threshold,
rate and dictionary identity is recorded and replayable. The pinned SCOWL
en_US Hunspell surface entries are bundled under their MIT/BSD terms, with
the full notice in `data/wordlists/LICENSE.en_US`. English vocabulary cannot
certify all Latin-script languages, technical notation or Greek hidden behind
Latin font encodings; qualification and rendered-page review determine whether
this profile is suitable for a corpus.

Successful OCR updates canonical Markdown, metrics, `text`, `ocr_success`,
`needs_ocr`, `extraction_mode="docling"`, and OCR artifact pointers/hashes in parquet,
then refreshes the OCR cleaner and score-only cleaner.

`mode="math_only"` continues to use existing Docling JSON. See
[math enrichment](../ocr_and_math_enhancement.md) and [migration](../ocr_plugin_migration.md).
