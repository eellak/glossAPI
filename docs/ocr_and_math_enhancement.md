# GPU OCR and Math Enrichment

This document summarizes how GlossAPI uses the GPU for OCR and formula/code enrichment, how to run each phase efficiently, and where artifacts are written.

## Overview

- Phase‑1 (Extract): PDF → Markdown via Docling; optional GPU OCR via RapidOCR (ONNXRuntime). Optionally emit JSON + formula index for Phase‑2.
- Phase‑2 (Enrich): From Docling JSON, decode math/code on the GPU (CodeFormula) and re‑emit enriched Markdown.

Backends
- `backend='rapidocr'` (default): Docling + RapidOCR; Phase‑2 math runs from Docling JSON.
- `backend='deepseek'`: DeepSeek‑OCR; equations are included inline in OCR output, so Phase‑2 math is not required and is treated as a no‑op.

Policy: never OCR and math on the same file
- If a file needs OCR, GlossAPI runs OCR only (no Phase‑2 on that file in the same pass).
- If a file does not need OCR but needs math, GlossAPI runs math‑only from Docling JSON. The JSON is produced by Phase‑1 (Docling layout) and must already exist.

### Python API layout

- DeepSeek entry point: `glossapi.ocr.deepseek.runner.run_for_files(...)`
- RapidOCR dispatcher: `glossapi.ocr.rapidocr.dispatch.run_via_extract(...)`
- Math enrichment: `glossapi.ocr.math.enrich.enrich_from_docling_json(...)`
- Utility helpers (Docling JSON / cleaning): `glossapi.ocr.utils.*`

## Prerequisites

- RapidOCR/Docling stack: `pip install '.[rapidocr]'`
- DeepSeek CLI stack (in a dedicated venv recommended): `pip install '.[deepseek]'`
- ONNXRuntime GPU installed (no CPU ORT): `onnxruntime-gpu==1.18.1`
- Torch CUDA installed: e.g., `torch==2.5.1+cu121`
- Packaged RapidOCR models/keys found under `glossapi/models/rapidocr/{onnx,keys}` or via `GLOSSAPI_RAPIDOCR_ONNX_DIR`.
- Optional helpers for Phase‑2 JSON: `pypdfium2`, `zstandard`.

Verify GPU readiness before forcing OCR or math:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"  # expects True, >=1
python -c "import onnxruntime as ort; print(ort.get_available_providers())"            # must include CUDAExecutionProvider
```

## Running Phase‑1 (Extract)

```python
from glossapi import Corpus
c = Corpus('IN','OUT')

# GPU OCR on PDFs; emit JSON + formula index for Phase‑2
c.extract(
    input_format='pdf',
    accel_type='CUDA',           # or use_gpus='multi' for multi‑GPU
    force_ocr=True,              # OCR always on for PDFs
    emit_formula_index=True,     # request json/<stem>.formula_index.jsonl alongside the default JSON
)
```

When `force_ocr=True` (or when math/code enrichment is enabled), GlossAPI automatically switches to the Docling backend and aborts if CUDA‑enabled torch/ONNXRuntime providers are not available.

Outputs:
- `markdown/<stem>.md`
- `json/<stem>.docling.json(.zst)` and `json/<stem>.formula_index.jsonl`
- `json/metrics/<stem>.metrics.json` and `json/metrics/<stem>.per_page.metrics.json`

## Running Phase‑2 (Enrich)

```python
from glossapi import Corpus
c = Corpus('OUT','OUT')  # same folder for both

# GPU formula/code decoding from JSON (writes enriched MD to markdown/<stem>.md)
c.formula_enrich_from_json(
    device='cuda',
    batch_size=12,       # tune for your GPU
)
```

Outputs:
- `markdown/<stem>.md` — enriched Markdown overwrites the plain MD
- `json/<stem>.latex_map.jsonl` — LaTeX strings + acceptance/metrics per item

## DeepSeek usage

Run OCR for files flagged by the cleaner as needing OCR (math flags are ignored for DeepSeek):

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='deepseek', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → runs OCR only for bad files; equations are included inline; Phase‑2 is skipped
```

If you need Phase‑2 math on files that do not require OCR, use RapidOCR/Docling and math‑only (expects Docling JSON from Phase‑1):

```python
c.ocr(backend='rapidocr', fix_bad=False, math_enhance=True, mode='math_only')
# → runs Phase‑2 on non‑OCR files only (requires Docling JSON)
```

## Multi‑GPU

Phase‑1 (extract):
```python
c.extract(input_format='pdf', use_gpus='multi', force_ocr=True)
```
Workers set `CUDA_VISIBLE_DEVICES` per process; Docling runs on `cuda:0` relative to each worker. OCR uses ORT GPU under the same process.

Phase‑2 (enrich):
```python
c.ocr(use_gpus='multi', math_batch_size=12)
```
Spawns math workers; each binds to its GPU using `CUDA_VISIBLE_DEVICES` and runs CodeFormula on `cuda:0` relative to that worker.

### Resuming and Recovering Workers

- `Corpus.ocr()` now persists end‑to‑end progress in the canonical parquet (`download_results/download_results.parquet`). As long as `reprocess_completed=False` (default) any row with `ocr_success=True` or `math_enriched=True` is skipped on the next run; pass `reprocess_completed=True` to force a redo or use the legacy alias `skip_existing=False`.
- Multi‑GPU math workers respawn automatically when a process crashes. Control the number of retries per GPU with `GLOSSAPI_MATH_RESPAWN_CAP` (default `5`). Active assignments are written to `logs/math_workers/gpu<N>.current` and the worker log directory can be overridden via `GLOSSAPI_WORKER_LOG_DIR`.
- When a GPU exceeds the respawn cap the remaining stems are added to the fatal skip‑list and copied to `downloads/problematic_math/` (PDFs) and `json/problematic_math/` (JSON artifacts) so they can be inspected or retried manually.
- Set `GLOSSAPI_WORKER_LOG_VERBOSE=0` to silence the per-worker binding banner; each worker still keeps an append-only log in the worker log directory for post‑mortem debugging.

## Performance & Tuning

- Batch sizes
  - Inline (Phase‑1): `GLOSSAPI_FORMULA_BATCH` (default 16) sets CodeFormula docling side throughput.
  - Phase‑2: `batch_size` / `math_batch_size` parameter (typ. 8–16) balances VRAM and speed.
- Images scale for OCR: `GLOSSAPI_IMAGES_SCALE` (~1.1–1.25) can improve detection on thin glyphs.
- CPU threads: cap `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to avoid CPU oversubscription on multi‑GPU nodes.

## Early‑stop & Post‑Processing Guards (Formula)

To keep LaTeX well‑formed and fast:
- Generation‑time (applied inside the decoder):
  - `GLOSSAPI_LATEX_EARLYSTOP` = `1|0` (default 1): enable early‑stop criteria for the HF generate path.
  - `GLOSSAPI_LATEX_MAX_CHARS` (default 3000): decoded‑length stop gate.
  - `GLOSSAPI_LATEX_MAX_REPEAT` (default 50): stop on excessive last‑token repetition.
  - `GLOSSAPI_LATEX_LEN_STRIDE` (default 16): decoding stride for the length check.
  - `GLOSSAPI_LATEX_MAX_NEW_TOKENS` (optional): cap new tokens at the decoder level (injected only if caller doesn’t specify one).

- Post‑processing (applied on the generated string):
  - `GLOSSAPI_LATEX_POST_ONLY_FAILED` = `1|0` (default 1): only sanitize when output looks pathological.
  - `GLOSSAPI_LATEX_POST_REPEAT_GATE` (default 50): consider output failed if the last token repeats more than this gate.
  - `GLOSSAPI_LATEX_POST_WINDDOWN` (default 12): clamp the repeated tail token to this run length.
  - `GLOSSAPI_LATEX_POST_MAX_CHARS` (default 3000): cap text length; prefers whitespace/`\` boundary.

The policy is centralized in `glossapi.text_sanitize`. Phase‑2 enrichment and the per‑page metrics both use the same sanitizer so counts and truncation flags are consistent.

## Artifact Placement Summary

```
OUT/
├── markdown/
│   └── <stem>.md                     # enriched Markdown (canonical)
├── json/
│   ├── <stem>.docling.json(.zst)
│   ├── <stem>.formula_index.jsonl
│   ├── <stem>.latex_map.jsonl
│   └── metrics/
│       ├── <stem>.metrics.json
│       └── <stem>.per_page.metrics.json
```

## Troubleshooting

- Missing CUDAExecutionProvider
  - Ensure `onnxruntime-gpu` is installed and `onnxruntime` CPU is uninstalled.
- Torch reports no CUDA
  - Check `nvidia-smi` and match Torch CUDA build to your driver.
- OCR is slow or falls back to CPU
  - Confirm ORT providers include CUDAExecutionProvider and that `accel_type='CUDA'` is used.
- Out of memory
  - Lower `batch_size` for Phase‑2, reduce `GLOSSAPI_IMAGES_SCALE`, or split inputs.
