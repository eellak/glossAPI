# Quickstart

This page shows the most common tasks in a few lines each. Every recipe assumes a source install of glossAPI in an
activated environment ([Onboarding Guide](getting_started.md)) and names the optional packages and hardware it needs.
Install optional packages in that environment with `python install_glossapi.py PACKAGE [--device DEVICE]`.
When you save a recipe as a script file, put the calls under `if __name__ == "__main__":`: per-document work runs in
worker processes started with `spawn`, which import the script again.

## Phase‑1 extraction profiles

### Stable (PyPDFium, size‑1)

Core install only; no GPU.

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf')  # safe PDFium backend, layout_v2 Markdown; OCR is off by default
```

This keeps Docling out of the hot path and is the recommended mode when you prioritise stability.

### Throughput (Docling, batched)

Needs the optional `layout` package (`python install_glossapi.py layout --device cuda126`, or `cuda132`) and a CUDA
GPU.

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA', phase1_backend='docling')
```

`phase1_backend='docling'` streams multiple PDFs through Docling’s converter and
should be used when you are comfortable trading some stability for throughput.

### Multi‑GPU

Needs the `layout` package and several CUDA GPUs.

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', use_gpus='multi', phase1_backend='docling')  # workers share a queued file list
```

Workers report per-batch summaries and extraction progress is persisted into
`download_results/download_results.parquet`, so you can restart multi-GPU runs
without losing progress (no extra checkpoint files required).

## OCR remediation (opt-in)

Needs the `ocr` package (`python install_glossapi.py ocr --device cpu`, or `mps` / `cuda126` / `cuda132`). The
default pinned Greek toolkit needs no options.

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf')
c.clean()
c.ocr(mode='ocr_bad')  # pinned Greek R8 + x1 toolkit; pages page quality routes (page_routing='all')
```

## Phase‑2 Math Enrichment (from JSON)

Needs the `layout` package (`python install_glossapi.py layout --device cuda126`, or `cuda132`) and a CUDA GPU.
Math enrichment consumes existing Docling JSON from an explicit Docling extraction. The safe PDFium backend does not
create these inputs.

```python
from glossapi import Corpus

c = Corpus('IN', 'OUT')
c.extract(
    input_format='pdf',
    phase1_backend='docling',
    accel_type='CUDA',
    export_doc_json=True,
    emit_formula_index=True,
)
c.formula_enrich_from_json(device='cuda', batch_size=12)  # enriched Markdown into OUT/markdown/<stem>.md
```

Progress (downloaded, OCRed, math-enriched) now lives in `download_results/download_results.parquet`; rerun `c.ocr(..., reprocess_completed=True)` whenever you need to force already successful rows back through OCR or math.

## Full Pipeline (download → extract → clean/ocr → section → annotate)

Needs the `download` package (`python install_glossapi.py download`) for `download()` and the `ocr` package for
`ocr()`; the other calls are core.

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.download(input_parquet='urls.parquet', url_column='url')  # optional: a parquet with a URL column
c.extract(input_format='pdf')        # Phase‑1 (no OCR by default)
c.clean()                            # compute quality; filter badness
c.ocr(mode='ocr_bad')                # default Greek R8+x1 plugin
c.section()                          # to parquet
c.annotate()                         # classify/annotate sections
```

See [OCR & Math Enrichment](ocr_and_math_enhancement.md) for OCR runtime details, batch sizes, and artifact
locations, and [OCR plugin migration](ocr_plugin_migration.md) for the toolkit pins, caches and the explicit
`ocr_options` of a custom plugin configuration.
