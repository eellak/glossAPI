# OCR remediation

Run extraction and cleaning first, then call
`corpus.ocr(mode="ocr_bad", ocr_options=plugin_options, device="cpu")`, where
`plugin_options` is configured by the external OCR package. An absent plugin
produces an explicit install/configuration error; no built-in OCR fallback is used.

The toolkit registers `glossapi = "greekocr.deploy.docling_plugin.docling_plugin"`
in the Docling `docling` entry-point group. Docling loads `ocr_engines()` with
`allow_external_plugins=True`; the engine owns its options, model files and inference.

`page_routing="flagged"` retains the cleaner's document gate (default).
`page_routing="all"` additionally admits unflagged documents with `ocr_pages` evidence.
Ranges are inclusive, one-based physical pages; a flagged document with no range
runs whole-document OCR. Skip-list and completed-document filtering still apply.
Partial OCR preserves native Markdown and replaces only selected page text where
page boundaries are known; ambiguous boundaries retain the original and append
an OCR supplement. The latest replacement per physical page is retained on reruns.

Successful OCR updates canonical Markdown, metrics, `text`, `ocr_success`,
`needs_ocr`, `extraction_mode="docling"`, and OCR artifact pointers/hashes in parquet,
then refreshes the OCR cleaner and score-only cleaner.

`mode="math_only"` continues to use existing Docling JSON. See
[math enrichment](../ocr_and_math_enhancement.md) and [migration](../ocr_plugin_migration.md).
