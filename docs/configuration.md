# Configuration & Environment Variables

This page lists the main knobs you can use to tune GlossAPI.

## GPU & Providers

- `CUDA_VISIBLE_DEVICES`: restrict/assign visible GPUs, e.g. `export CUDA_VISIBLE_DEVICES=0,1`.
- `GLOSSAPI_DOCLING_DEVICE`: preferred device for Docling (inside a worker), e.g. `cuda:0`.

## OCR & Parsing

- `GLOSSAPI_IMAGES_SCALE`: image scale hint for OCR/layout (default ~1.1–1.25).
- `GLOSSAPI_FORMULA_BATCH`: inline CodeFormula batch size (default `16`).

### Batch Policy & PDF Backend

GlossAPI exposes two Phase‑1 profiles. Use `Corpus.extract(..., phase1_backend='docling')` to switch from the default safe backend. The legacy environment variables `GLOSSAPI_BATCH_POLICY` and `GLOSSAPI_BATCH_MAX` are still parsed for backward compatibility but emit a deprecation warning and will be removed in a future release.

Regardless of backend, the extractor clamps OMP/OpenBLAS/MKL pools to one thread per worker so multi‑GPU runs do not explode thread counts.

### DeepSeek optional dependencies

Install DeepSeek backend extras to enable the DeepSeek OCR path (imports remain lazy, so the package is optional). Use the CUDA 12.1 wheels for both vLLM and Torch:

```bash
pip install '.[deepseek]'

# Install Torch CUDA 12.1 wheels (required by the DeepSeek script)
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121'

# Alternatively, use the requirements file (edit to uncomment torch lines):
pip install -r deepseek-ocr/requirements-deepseek.txt
```

When using `backend='deepseek'`, equations are included inline in the OCR output; Phase‑2 math flags are accepted but skipped.

## Math Enrichment (Phase‑2)

- `GLOSSAPI_LATEX_EARLYSTOP` = `1|0` (default 1): enable/disable early‑stop wrapper.
- `GLOSSAPI_LATEX_MAX_CHARS` (default `3000`): cap decoded LaTeX length.
- `GLOSSAPI_LATEX_MAX_REPEAT` (default `50`): stop on last‑token repetition runs.
- `GLOSSAPI_LATEX_MAX_NEW_TOKENS` (optional): cap decoder new tokens.
- `GLOSSAPI_LATEX_LEN_STRIDE` (default `16`): stride for length checks.
- `GLOSSAPI_MATH_RESPAWN_CAP` (default `5`): maximum number of times a crashed math worker is respawned per GPU during multi‑GPU enrichment (set to `0` to disable respawns).

### Centralized LaTeX Policy (Post‑processing)

- `GLOSSAPI_LATEX_POST_ONLY_FAILED` = `1|0` (default `1`): only sanitize when output looks problematic.
- `GLOSSAPI_LATEX_POST_REPEAT_GATE` (default `50`): consider output failed if tail token repeats more than this.
- `GLOSSAPI_LATEX_POST_WINDDOWN` (default `12`): clamp repeated tail token to this run length.
- `GLOSSAPI_LATEX_POST_MAX_CHARS` (default `3000`): cap text length (prefers whitespace/`\` boundary).

All LaTeX policy knobs are loaded via `glossapi.text_sanitize.load_latex_policy()` and used consistently in early‑stop, Phase‑2 post‑processing, and metrics.

## Performance & Caches

- `OMP_NUM_THREADS` / `MKL_NUM_THREADS`: cap CPU threads to avoid oversubscription.
- Cache locations: `HF_HOME`, `XDG_CACHE_HOME`, `DOCLING_CACHE_DIR`.

## Worker Logging

- `GLOSSAPI_WORKER_LOG_DIR`: override the directory used for per-worker logs and `gpu<N>.current` markers (defaults to `logs/ocr_workers/` or `logs/math_workers/` under the output directory).
- `GLOSSAPI_WORKER_LOG_VERBOSE` = `1|0` (default `1`): emit (or suppress) the GPU binding banner each worker prints on startup.

## RapidOCR Model Paths

- `GLOSSAPI_RAPIDOCR_ONNX_DIR`: directory containing `det/rec/cls` ONNX models and keys.

## Triage & Parquet

- Triage always writes both:
  - Sidecar summaries: `sidecars/triage/{stem}.json` (per document)
  - Parquet updates: `download_results/download_results.parquet` (adds/updates rows)
- Default recommendation policy: enrich if `formula_total > 0` (skip only no‑math docs).
- Legacy heuristic (p90/pages thresholds) can be enabled with `GLOSSAPI_TRIAGE_HEURISTIC=1`.
