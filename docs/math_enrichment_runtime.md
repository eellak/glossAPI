# Math Enrichment Runtime Guide (Early‑Stop + Post‑Processing)

This guide documents how math/code enrichment runs in two phases, how the generation early‑stop logic works, and how the optional post‑processing policy applies only to problematic outputs. It also shows how to run targeted Phase‑2 on a few items instead of the entire document.

## Policy Summary

- JSON is an intermediate only when enhancements are used (math/code). Standard runs stay PDF → MD.
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

Emit JSON only when enhancements are used

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
# Phase‑1 (layout only); enable JSON export when you plan to enrich later
c.extract(input_format='pdf', use_gpus='multi', export_doc_json=True, emit_formula_index=True)
```

OCR + math enrichment in one call (JSON ensured automatically)

```python
from glossapi import Corpus
c = Corpus('OUT','OUT')
# Re‑extract with OCR on GPU and immediately enrich from JSON
c.ocr(math_enhance=True)
```

Targeted math (only specific items)

```python
from glossapi import Corpus
c = Corpus('OUT','OUT')
# Only page/index pairs you want (page_no is 1‑based; item_index is per‑page occurrence)
targets = { 'JSM_564': [(14,1), (23,2)] }
c.ocr(math_enhance=True, math_targets=targets, math_batch_size=4)
```

## Targeted Phase‑2 From CLI

- Probe early‑stop status:
  - `python scripts/probe_formula_earlystop.py`
- Target the 5 longest formulas in a run folder (uses Corpus under the hood):
  - `python scripts/phase2_target_longest.py --run-dir OUTS/phase1_math_json_* --device cuda --batch-size 4`
  - Writes summary to `json/top5_longest_formulas.jsonl` and re‑runs Phase‑2 for those items only.

## OCR/Model Constraints (recap)

- ORT GPU only: uninstall `onnxruntime` CPU; use `onnxruntime-gpu`.
- RapidOCR keys: Docling 2.48.0 needs `Rec.rec_keys_path` patch (see README).
- Model discovery: set `GLOSSAPI_RAPIDOCR_ONNX_DIR` or package models under `glossapi/models/rapidocr/`.
- Optional Torch CUDA: needed for GPU layout/enrichment; see README for the CUDA wheels.

## Multi‑GPU

- Extract: uses a shared work queue across all visible GPUs (`use_gpus='multi'`).
- OCR/Phase‑2: runs on GPU; integrated targeted processing supports small batches. Roadmap: multi‑GPU queue for OCR as well.

## Troubleshooting

- If outputs still look repetitive: tighten early‑stop (`MAX_REPEAT`, `MAX_CHARS`) and rerun targeted Phase‑2.
- To observe cleanup: enable post‑processing flags and inspect `*.latex_map.jsonl` (`post_applied`, `truncated_by_*`, `tail_run`).

