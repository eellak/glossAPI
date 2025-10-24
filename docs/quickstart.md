# Quickstart

This page shows the most common tasks in a few lines each.

## Phase‑1 extraction profiles

### Stable (PyPDFium, size‑1)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA')  # OCR is off by default
```

This keeps Docling’s native parser out of the hot path and is the recommended
mode when you prioritise stability.

### Throughput (Docling, batched)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA', phase1_backend='docling')
```

`phase1_backend='docling'` streams multiple PDFs through Docling’s converter and
should be used when you are comfortable trading some stability for throughput.

### Multi‑GPU

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', use_gpus='multi')  # workers share a queued file list
```

Workers report per-batch summaries and the controller persists a single
`.processing_state.pkl`, so you can restart multi-GPU runs without losing
progress.

## GPU OCR (opt-in)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA', force_ocr=True)
# or reuse multi-GPU batching
c.extract(input_format='pdf', use_gpus='multi', force_ocr=True)
```

## Phase‑2 Math Enrichment (from JSON)

```python
from glossapi import Corpus
c = Corpus('OUT', 'OUT')  # same folder for input/output is fine

# Emit JSON/indices first (JSON now defaults on; request the index explicitly)
c.extract(input_format='pdf', accel_type='CUDA', emit_formula_index=True)

# Enrich math/code on GPU and write enriched Markdown into markdown/<stem>.md
c.formula_enrich_from_json(device='cuda', batch_size=12)
```

Progress (downloaded, OCRed, math-enriched) now lives in `download_results/download_results.parquet`; rerun `c.ocr(..., reprocess_completed=True)` whenever you need to force already successful rows back through OCR or math.

## Full Pipeline (download → extract → clean/ocr → section → annotate)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.download(url_column='url')         # optional, if you have URLs parquet
c.extract(input_format='pdf')        # Phase‑1 (no OCR by default)
c.clean()                            # compute quality; filter badness
c.ocr()                              # re‑extract bad ones and enrich math/code
c.section()                          # to parquet
c.annotate()                         # classify/annotate sections
```

See ocr_and_math_enhancement.md for GPU details, batch sizes, and artifact locations.
