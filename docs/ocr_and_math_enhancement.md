# GPU OCR and Math Enrichment

This document summarizes how GlossAPI uses the GPU for OCR remediation and formula/code enrichment, how to run each phase efficiently, and where artifacts are written.

## Overview

- Phase‑1 (Extract): PDF → Markdown via Docling or the safe backend. Optionally emit JSON + formula index for Phase‑2.
- Phase‑2 (Enrich): From Docling JSON, decode math/code on the GPU (CodeFormula) and re‑emit enriched Markdown.

Backends
- `backend='deepseek'`: DeepSeek-OCR-2; equations are included inline in OCR output, so Phase‑2 math is not required and is treated as a no‑op.

Policy: never OCR and math on the same file
- If a file needs OCR, GlossAPI runs OCR only (no Phase‑2 on that file in the same pass).
- If a file does not need OCR but needs math, GlossAPI runs math‑only from Docling JSON. The JSON is produced by Phase‑1 (Docling layout) and must already exist.

### Python API layout

- DeepSeek entry point: `glossapi.ocr.deepseek.runner.run_for_files(...)`
- Math enrichment: `glossapi.ocr.math.enrich.enrich_from_docling_json(...)`
- Utility helpers (Docling JSON / cleaning): `glossapi.ocr.utils.*`

## Prerequisites

- Main GlossAPI stack: `./dependency_setup/setup_glossapi.sh --mode docling`
- DeepSeek runtime: `./dependency_setup/setup_deepseek_uv.sh --venv dependency_setup/.venvs/deepseek`
- Torch CUDA installed in the DeepSeek env (the uv setup pins the tested stack).
- Optional helpers for Phase‑2 JSON: `pypdfium2`, `zstandard`.

### Standard DeepSeek venv

Use a dedicated OCR runtime and treat it as the source of truth for DeepSeek runs:

```bash
./dependency_setup/setup_deepseek_uv.sh \
  --venv dependency_setup/.venvs/deepseek \
  --model-root /path/to/deepseek-ocr-2-model \
  --download-model \
  --run-tests --smoke-test
```

Recommended environment variables after setup:

```bash
export GLOSSAPI_DEEPSEEK_ALLOW_CLI=1
export GLOSSAPI_DEEPSEEK_ALLOW_STUB=0
export GLOSSAPI_DEEPSEEK_PYTHON="$PWD/dependency_setup/.venvs/deepseek/bin/python"
export GLOSSAPI_DEEPSEEK_MODEL_DIR="/path/to/deepseek-ocr-2-model/DeepSeek-OCR-2"
```

The OCR runtime should not silently drift between ad hoc virtual environments during benchmarking. If a benchmark uses a different DeepSeek venv, treat the result as a different runtime stack.

For the standardized OA-style single-GPU benchmark method and exactness check,
see [DeepSeek Single-GPU Benchmarking](operations/deepseek_single_gpu_benchmarking.md).

Verify GPU readiness before forcing OCR or math:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"  # expects True, >=1
```

## Running Phase‑1 (Extract)

```python
from glossapi import Corpus
c = Corpus('IN','OUT')

# Emit JSON + formula index for Phase‑2
c.extract(
    input_format='pdf',
    accel_type='CUDA',
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

## DeepSeek usage

Run OCR for files flagged by the cleaner as needing OCR (math flags are ignored for DeepSeek):

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.ocr(backend='deepseek', fix_bad=True, math_enhance=True, mode='ocr_bad_then_math')
# → runs OCR only for bad files; equations are included inline; Phase‑2 is skipped
```

If you need Phase‑2 math on files that do not require OCR, run `math_only` after Docling extraction with JSON enabled.

### DeepSeek fast path

The current recommended high-throughput DeepSeek configuration is:

- `runtime_backend='vllm'`
- `ocr_profile='markdown_grounded'`
- `max_new_tokens=2048` as the standard default ceiling
- `repair_mode='auto'` to keep markdown as the primary output while selectively rerunning suspicious pages
- `scheduler='auto'` so multi-GPU vLLM runs resolve to exact-fill page-range batching
- `target_batch_pages=160`
- large `vllm_batch_size` chosen to keep `sec/page/GPU` at or below the best validated floor for the target hardware

Example:

```python
c.ocr(
    backend='deepseek',
    fix_bad=True,
    math_enhance=False,
    runtime_backend='vllm',
    ocr_profile='markdown_grounded',
    max_new_tokens=2048,
    vllm_batch_size=160,
    gpu_memory_utilization=0.9,
    repair_mode='auto',
    scheduler='auto',
    target_batch_pages=160,
    use_gpus='multi',
)
```

`repair_mode='auto'` runs the pipeline in distinct phases inside the vLLM runner:

1. markdown first pass over all rendered pages
2. cheap per-page triage using output quality plus simple image density statistics
3. plain-text rerun bucket for garbage markdown pages
4. tiled markdown rerun bucket for short coverage failures

This keeps the fast path batched while avoiding per-page sequential fallback overhead.

### What is now implemented

- Empty-page skipping before OCR dispatch
- Streaming garbage early-stop during markdown generation
- Plain-text retry for pages that hit the garbage early-stop
- Multi-GPU exact-fill page-range scheduling for the DeepSeek runner
- Benchmark harness support for `whole_doc`, `fixed_shard`, and `exact_fill`
- Corpus API forwarding for the scheduler controls

## Multi‑GPU

Phase‑1 (extract):
```python
c.extract(input_format='pdf', use_gpus='multi', phase1_backend='docling', workers_per_device=2)
```
Workers set `CUDA_VISIBLE_DEVICES` per process; Docling runs on `cuda:0` relative to each worker.

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

### Validated benchmark floor

The current non-regression metric is `sec/page/GPU`.

Validated on 2026-03-30:

- Host: AWS `g7e.48xlarge`
- Runtime: `vllm`
- Profile: `markdown_grounded`
- Render DPI: `144`
- GPU memory utilization: `0.9`
- Best large-batch single-GPU floor observed: `0.3109 sec/page/GPU`

Production markdown+repair benchmark on the same host:

- Corpus: `43` OA PDFs, `7,624` pages
- Runtime: `vllm`
- Profile: `markdown_grounded`
- Repair mode: `auto`
- Max new tokens: `2048`
- GPUs: `8`
- Static sharding (`1` shard/GPU), validated rerun after classifier hardening: `558.88s` wall, `0.0733 sec/page` overall, `0.4912` to `0.5475 sec/page/GPU`
- Streaming admission (`stream_batch_pages=160`): `928.81s` wall, `0.1218 sec/page` overall, `0.5469` to `0.6856 sec/page/GPU`
- Peak VRAM in both runs stayed at about `88,953 MiB` per active GPU
- Static active-lane GPU utilization averaged about `65%` to `75%`; streaming active-lane utilization stayed similar while whole-run occupancy got worse because more lanes sat idle between batches

Validated on 2026-03-31 after standardizing the DeepSeek runtime ceiling back to `2048` and restoring the persistent one-process-per-lane architecture:

- Corpus: `43` OA PDFs, `7,624` pages
- Runtime: `vllm`
- Profile: `markdown_grounded`
- Repair mode: `auto`
- Scheduler: `whole_doc`
- Max new tokens: `2048`
- GPUs: `8`
- Clean rebuilt whole-document rerun: about `541s` wall, `0.0710 sec/page` overall, and `0.3927` to `0.5000 sec/page/GPU`

Interpretation:

- The rebuilt stack is back near the validated March 30 throughput once the silent `8192` ceiling regression is removed.
- The remaining performance problem is not raw inference speed; it is whole-document tail imbalance, where one oversized PDF can keep a single GPU busy after the other lanes finish.
- Multi-GPU `exact_fill` must therefore be benchmarked only on the persistent lane-worker architecture. The earlier exact-fill regression was caused by spawning a fresh OCR CLI per batch, not by the scheduling idea itself.

Decision:

- Keep static sharding as the default large-run pipeline shape for now
- Do not enable streaming admission by default yet; on this benchmark it regressed badly versus static sharding
- Treat the earlier `0.3109 sec/page/GPU` result as the raw floor, and the static repaired-markdown result above as the current production-like baseline on this hardware
- Treat the 2026-03-31 clean whole-document rerun as the restored benchmark sanity check for the standardized `2048` ceiling on the rebuilt runtime

Attention/runtime note:

- The production fast path is `vllm`; logs on this stack show `flashinfer` autotuning plus CUDA graph capture
- Transformers remain the fallback path; prefer `flash_attention_2` there and do not optimize around `sdpa`

That number is the floor to preserve or beat when tuning the full markdown pipeline. Faster raw runs that change the effective output mode or bypass repair logic do not replace it as the production baseline.

Default policy note:

- The standard DeepSeek OCR default is now `max_new_tokens=2048` for both the Transformers and vLLM runners.
- Leaving the flag unset must not silently expand to a larger ceiling such as `8192`.
- When comparing benchmark runs, treat a different token ceiling or a different DeepSeek venv as a different runtime/configuration.

- Batch sizes
  - Inline (Phase‑1): `GLOSSAPI_FORMULA_BATCH` (default 16) sets CodeFormula throughput.
  - Phase‑2: `batch_size` / `math_batch_size` parameter (typ. 8–16) balances VRAM and speed.
  - DeepSeek vLLM: push `vllm_batch_size` as high as the hardware allows while tracking `sec/page/GPU`; on the validated `g7e.48xlarge` path, larger batches continued improving throughput through `batch_size=160`.
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

- Torch reports no CUDA
  - Check `nvidia-smi` and match Torch CUDA build to your driver.
- Out of memory
  - Lower `batch_size` for Phase‑2, reduce `GLOSSAPI_IMAGES_SCALE`, or split inputs.
