# Configuration & Environment Variables

This page lists the main knobs you can use to tune GlossAPI.

## GPU & Providers

- `CUDA_VISIBLE_DEVICES`: restrict/assign visible GPUs, e.g. `export CUDA_VISIBLE_DEVICES=0,1`.
- `GLOSSAPI_DOCLING_DEVICE`: preferred device for Docling (inside a worker), e.g. `cuda:0`.

## OCR & Parsing

- `GLOSSAPI_IMAGES_SCALE`: image scale hint for OCR/layout (default ~1.1–1.25).
- `GLOSSAPI_FORMULA_BATCH`: inline CodeFormula batch size (default `16`).

## Math Enrichment (Phase‑2)

- `GLOSSAPI_LATEX_EARLYSTOP` = `1|0` (default 1): enable/disable early‑stop wrapper.
- `GLOSSAPI_LATEX_MAX_CHARS` (default `3000`): cap decoded LaTeX length.
- `GLOSSAPI_LATEX_MAX_REPEAT` (default `50`): stop on last‑token repetition runs.
- `GLOSSAPI_LATEX_MAX_NEW_TOKENS` (optional): cap decoder new tokens.
- `GLOSSAPI_LATEX_LEN_STRIDE` (default `16`): stride for length checks.

## Performance & Caches

- `OMP_NUM_THREADS` / `MKL_NUM_THREADS`: cap CPU threads to avoid oversubscription.
- Cache locations: `HF_HOME`, `XDG_CACHE_HOME`, `DOCLING_CACHE_DIR`.

## RapidOCR Model Paths

- `GLOSSAPI_RAPIDOCR_ONNX_DIR`: directory containing `det/rec/cls` ONNX models and keys.

## Parquet Sidecars

- `GLOSSAPI_PARQUET_COMPACTOR` = `1|0` (default 1): sidecars for triage/math; legacy mode mutates parquet in place.
