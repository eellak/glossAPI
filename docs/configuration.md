# Configuration & Environment Variables

This page lists the main knobs you can use to tune GlossAPI.

## GPU & Providers

- `CUDA_VISIBLE_DEVICES`: restrict/assign visible GPUs, e.g. `export CUDA_VISIBLE_DEVICES=0,1`.
- `GLOSSAPI_DOCLING_DEVICE`: preferred device for Docling (inside a worker), e.g. `cuda:0`.

## OCR & Parsing

- `GLOSSAPI_IMAGES_SCALE`: image scale hint for OCR/layout (default ~1.1–1.25).
- `GLOSSAPI_FORMULA_BATCH`: inline CodeFormula batch size (default `16`).
- `GLOSSAPI_IMPORT_TORCH`: when set to `1|true|yes`, force on-demand Torch import inside workers (useful when Torch is optional in the environment).

### Batch Policy & PDF Backend

GlossAPI now ships with two Phase‑1 extraction profiles:

- **Safe (default)** — size‑1 batching with the PyPDFium backend. This keeps Docling’s native parser out of the hot path and is the recommended mode when stability matters most.
- **Docling/native** — larger batches using `docling_parse`. Faster on clean corpora, but reintroduces the concurrency risk highlighted in the core dump analysis.

Control the profile with these knobs:

- `GLOSSAPI_BATCH_POLICY`
  - `safe` *(default)* → PyPDFium backend, forces `max_batch_files=1` unless overridden programmatically
  - `pypdfium` → explicit PyPDFium request even if you raise the batch size
  - `docling` (or any other value) → keep the native Docling backend
- `GLOSSAPI_BATCH_MAX` (default `3`)
  - Maximum number of documents passed to `convert_all` when the policy allows batching.
- `configure_batch_policy(policy, max_batch_files=None, prefer_safe_backend=None)` — programmatic override on `GlossExtract`/`Corpus.extractor`.

Regardless of policy, the extractor clamps OMP/OpenBLAS/MKL pools to one thread per worker so multi‑GPU runs do not explode thread counts.

## Math Enrichment (Phase‑2)

- `GLOSSAPI_LATEX_EARLYSTOP` = `1|0` (default 1): enable/disable early‑stop wrapper.
- `GLOSSAPI_LATEX_MAX_CHARS` (default `3000`): cap decoded LaTeX length.
- `GLOSSAPI_LATEX_MAX_REPEAT` (default `50`): stop on last‑token repetition runs.
- `GLOSSAPI_LATEX_MAX_NEW_TOKENS` (optional): cap decoder new tokens.
- `GLOSSAPI_LATEX_LEN_STRIDE` (default `16`): stride for length checks.

### Centralized LaTeX Policy (Post‑processing)

- `GLOSSAPI_LATEX_POST_ONLY_FAILED` = `1|0` (default `1`): only sanitize when output looks problematic.
- `GLOSSAPI_LATEX_POST_REPEAT_GATE` (default `50`): consider output failed if tail token repeats more than this.
- `GLOSSAPI_LATEX_POST_WINDDOWN` (default `12`): clamp repeated tail token to this run length.
- `GLOSSAPI_LATEX_POST_MAX_CHARS` (default `3000`): cap text length (prefers whitespace/`\` boundary).

All LaTeX policy knobs are loaded via `glossapi.text_sanitize.load_latex_policy()` and used consistently in early‑stop, Phase‑2 post‑processing, and metrics.

## Performance & Caches

- `OMP_NUM_THREADS` / `MKL_NUM_THREADS`: cap CPU threads to avoid oversubscription.
- Cache locations: `HF_HOME`, `XDG_CACHE_HOME`, `DOCLING_CACHE_DIR`.

## RapidOCR Model Paths

- `GLOSSAPI_RAPIDOCR_ONNX_DIR`: directory containing `det/rec/cls` ONNX models and keys.

## Triage & Parquet

- Triage always writes both:
  - Sidecar summaries: `sidecars/triage/{stem}.json` (per document)
  - Parquet updates: `download_results/download_results.parquet` (adds/updates rows)
- Default recommendation policy: enrich if `formula_total > 0` (skip only no‑math docs).
- Legacy heuristic (p90/pages thresholds) can be enabled with `GLOSSAPI_TRIAGE_HEURISTIC=1`.
