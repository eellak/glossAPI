# Quickstart

This page shows the most common tasks in a few lines each.

## Phase‑1 extraction (layout only, single GPU)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA')  # OCR is off by default
```

## Phase‑1 extraction (layout only, multi‑GPU)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', use_gpus='multi')  # workers share a queued file list
```

Workers now report per-batch summaries and the controller persists a single
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
