# OCR plugin migration — 28 October 2026

DeepSeek OCR, its `deepseek` install extra, runtime setup, CLI scripts and serving
controls have been removed. The in-repository C1 OCR plugin and its codec are also
removed. Install and configure the toolkit's `greekocr` package instead; GlossAPI
no longer ships an OCR engine or OCR weights. The unused bundled RapidOCR ONNX
models and ONNX Runtime provider diagnostics have been removed.

External engines register under Docling's **`docling`** entry-point group. The
toolkit registers:

```toml
[project.entry-points."docling"]
glossapi = "greekocr.deploy.docling_plugin.docling_plugin"
```

The entry-point module supplies `ocr_engines()`; Docling loads the engine with
`allow_external_plugins=True`. Obtain configured options from the installed
plugin, including its model/checkpoint and routing configuration, then pass them:

```python
# plugin_options is an options instance configured using the external package.
corpus.ocr(mode="ocr_bad", ocr_options=plugin_options, device="cpu")
# Or opt into OCR during sequential layout extraction:
corpus.extract(phase1_backend="docling", accel_type="CPU",
               batched=False, ocr_options=plugin_options)
```

Without options, OCR remediation reports how to install/configure a plugin.
The old `ocr_engine`, `ocr_checkpoint`, `ocr_codec` and related C1 extraction
arguments are replaced by `ocr_options`. DeepSeek-only inference, repair, runtime
and scheduling arguments have no replacement in GlossAPI; configure the external
engine through its own options. OCR remediation currently runs sequentially on
`device`; `use_gpus`/`devices` control the separate math enrichment path.

`ocr_pages`, inclusive physical page ranges, cleaner routing decisions, skip-list
and completed-file filtering remain backend-independent. Partial OCR retains
native Markdown and previously replaced pages. Math enrichment remains available
through the Docling layout package, including `mode="math_only"` without an OCR
plugin. Select `mode="ocr_bad"` for OCR alone; `ocr_bad_then_math` also enriches
non-OCR documents with existing JSON.

The external toolkit's version compatibility and real-model output quality are
qualified by the OCR integration task; removing the old engines does not establish
that compatibility.
