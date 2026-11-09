# Math Enrichment Runtime Guide (Early‑Stop + Post‑Processing)

This guide documents how math/code enrichment runs in two phases, how the generation early‑stop logic works, and how the optional post‑processing policy applies only to problematic outputs. It also shows how to run targeted Phase‑2 on a few items instead of the entire document.

## Policy Summary

- JSON is an intermediate only when enhancements are used (math/code). Standard runs stay PDF → MD.
- Math enrichment consumes existing Docling JSON from an explicit Docling extraction. The safe PDFium backend does not
  create these inputs. It needs the optional `layout` package (`python install_glossapi.py layout --device cuda126`).
- Early‑stop runs during decoding to prevent runaway outputs with minimal overhead.
- Post‑processing runs only on clearly failed cases to “wind down” degenerate runs; otherwise the generated LaTeX is left untouched.
- Both phases run on GPU by default; CPU layout remains an option.

## Environment Variables

Early‑stop (applied during decoding; default enabled)
- `GLOSSAPI_LATEX_EARLYSTOP` (default `1`): enable/disable wrapper.
- `GLOSSAPI_LATEX_MAX_CHARS` (default `3000`): char cap during generation.
- `GLOSSAPI_LATEX_MAX_REPEAT` (default `50`): stop if the last token repeats more than N times.
- `GLOSSAPI_LATEX_MAX_NEW_TOKENS` (optional): hard token cap for the decoder.
- `GLOSSAPI_LATEX_LEN_STRIDE` (default `16`): check frequency (every N steps) for char length.

Post‑processing (applied after decoding; default only on failed cases)
- `GLOSSAPI_LATEX_POST_ONLY_FAILED` (default `1`): apply post‑processing only if gating triggers.
- `GLOSSAPI_LATEX_POST_REPEAT_GATE` (default `50`): treat as failed if tail run > gate.
- `GLOSSAPI_LATEX_POST_WINDDOWN` (default `12`): wind down tail run to this count.
- `GLOSSAPI_LATEX_POST_MAX_CHARS` (default `3000`): treat as failed if len > cap and trim to cap.

Notes
- Early‑stop saves GPU time by stopping the model; post‑processing is a last‑resort cleanup.
- You can tighten early‑stop to test effects quickly: e.g., `MAX_CHARS=1200`, `MAX_REPEAT=30`.

## Integrated Pipeline

Phase‑1 with the Docling backend writes the Docling JSON (`export_doc_json=True`, the default, applies to the Docling
backend only); request the formula index when you plan to enrich later:

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
c.formula_enrich_from_json(device='cuda', batch_size=12)
```

Math enrichment through `Corpus.ocr()`: `mode='math_only', fix_bad=False` enriches the documents that already have
Docling JSON and performs no OCR. Without a mode, `c.ocr()` selects `ocr_bad_then_math`: it OCRs the routed pages
first and then enriches only the documents OCR did not touch, from their existing Docling JSON. No call creates
Docling JSON for you.

```python
from glossapi import Corpus
c = Corpus('OUT', 'OUT')
c.ocr(mode='math_only', fix_bad=False, math_batch_size=12)
```

Targeted math (only specific items). Pass the stems as `files` as well: `targets_by_stem` alone restricts the items
inside the named stems but does not restrict which discovered documents are enriched.

```python
from glossapi import Corpus
c = Corpus('OUT', 'OUT')
# Only page/index pairs you want (page_no is 1‑based; item_index is per‑page occurrence)
targets = {'JSM_564': [(14, 1), (23, 2)]}
c.formula_enrich_from_json(files=list(targets), targets_by_stem=targets, device='cuda', batch_size=4)
```

These recipes match the current signatures; real Docling math inference is not part of the CPU test suite and needs
a CUDA host with the `layout` package.

## OCR/Model Constraints (recap)

- Optional Torch CUDA: needed for GPU layout/enrichment; see README for the CUDA wheels.

## Multi‑GPU

- Extract: uses a shared work queue across all visible GPUs (`use_gpus='multi'`).
- Phase‑2 math: `Corpus.ocr(mode='math_only', fix_bad=False, use_gpus='multi')` spreads math enrichment over the visible GPUs; targeted processing supports small batches.
- OCR: the external plugin runs on one device (see [OCR plugin migration](ocr_plugin_migration.md)). Multi-GPU OCR is a target that has not been qualified.

## Troubleshooting

- If outputs still look repetitive: tighten early‑stop (`MAX_REPEAT`, `MAX_CHARS`) and rerun targeted Phase‑2 with `Corpus.formula_enrich_from_json(files=list(targets), targets_by_stem=targets)`.
- To observe cleanup: enable post‑processing flags and inspect `*.latex_map.jsonl` (`post_applied`, `truncated_by_*`, `tail_run`).
