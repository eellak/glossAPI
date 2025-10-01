# GPU OCR and Math Enrichment

This document summarizes how GlossAPI uses the GPU for OCR and formula/code enrichment, how to run each phase efficiently, and where artifacts are written.

## Overview

- Phase‑1 (Extract): PDF → Markdown via Docling; optional GPU OCR via RapidOCR (ONNXRuntime). Optionally emit JSON + formula index for Phase‑2.
- Phase‑2 (Enrich): From Docling JSON, decode math/code on the GPU (CodeFormula) and re‑emit enriched Markdown.

## Prerequisites

- ONNXRuntime GPU installed (no CPU ORT): `onnxruntime-gpu==1.18.1`
- Torch CUDA installed: e.g., `torch==2.5.1+cu121`
- Packaged RapidOCR models/keys found under `glossapi/models/rapidocr/{onnx,keys}` or via `GLOSSAPI_RAPIDOCR_ONNX_DIR`.
- Optional helpers for Phase‑2 JSON: `pypdfium2`, `zstandard`.

Always set the runtime environment before forcing OCR or math:

```bash
export GLOSSAPI_BATCH_POLICY=docling
export GLOSSAPI_IMPORT_TORCH=1
# optional: limit visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # must list CUDAExecutionProvider
```

Check:
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # must include CUDAExecutionProvider
python -c "import torch; import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
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
