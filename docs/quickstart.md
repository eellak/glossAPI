# Quickstart

This page shows the most common tasks in a few lines each.

## GPU OCR (single GPU)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', accel_type='CUDA', force_ocr=True)
```

## GPU OCR (multi‑GPU)

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', use_gpus='multi', force_ocr=True)
```

## Phase‑2 Math Enrichment (from JSON)

```python
from glossapi import Corpus
c = Corpus('OUT', 'OUT')  # same folder for input/output is fine

# Emit JSON/indices first (if you didn’t already)
c.extract(input_format='pdf', accel_type='CUDA', export_doc_json=True, emit_formula_index=True)

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
c.ocr(force=True)                    # re‑extract only bad ones with GPU OCR
c.section()                          # to parquet
c.annotate()                         # classify/annotate sections
```

See ocr_and_math_enhancement.md for GPU details, batch sizes, and artifact locations.
