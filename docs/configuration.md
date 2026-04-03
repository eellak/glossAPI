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

### Docling Runtime Tuning

These optional knobs map directly to current Docling `PdfPipelineOptions` fields and are mainly useful for benchmarking on strong GPUs:

- `GLOSSAPI_DOCLING_MAX_BATCH_FILES`: override the number of PDF documents a single Phase‑1 Docling worker processes per extractor batch. Defaults to `1` in GlossAPI for stability; raise it deliberately when benchmarking fresh A100 nodes.
- `GLOSSAPI_DOCLING_BATCH_TARGET_PAGES`: target page budget for each queued multi‑GPU Docling work item. Defaults to `256`; lower it when a single worker hoards long PDFs, raise it when a strong GPU can keep larger mixed bundles resident.
- `GLOSSAPI_DOCLING_LAYOUT_BATCH_SIZE`: override Docling `layout_batch_size`.
- `GLOSSAPI_DOCLING_TABLE_BATCH_SIZE`: override Docling `table_batch_size`.
- `GLOSSAPI_DOCLING_OCR_BATCH_SIZE`: override Docling `ocr_batch_size` even though Phase‑1 OCR stays disabled.
- `GLOSSAPI_DOCLING_QUEUE_MAX_SIZE`: override Docling `queue_max_size`.
- `GLOSSAPI_DOCLING_DOCUMENT_TIMEOUT`: override Docling `document_timeout`.
- `GLOSSAPI_DOCLING_BATCH_POLL_INTERVAL`: override Docling `batch_polling_interval_seconds`.

### DeepSeek optional dependencies

Install DeepSeek backend extras to enable the DeepSeek OCR path. The recommended path is the dedicated `uv` environment:

```bash
./dependency_setup/setup_deepseek_uv.sh --venv dependency_setup/.venvs/deepseek
```

When using `backend='deepseek'`, equations are included inline in the OCR output; Phase‑2 math flags are accepted but skipped.
The dedicated uv profile is OCR-only and does not install the Docling extraction stack.

### DeepSeek runtime controls

- `GLOSSAPI_DEEPSEEK_ALLOW_STUB`: must remain `0`; stub execution is rejected.
- `GLOSSAPI_DEEPSEEK_ALLOW_CLI`: keep at `1` to require the real runtime.
- `GLOSSAPI_DEEPSEEK_PYTHON`: absolute path to the Python interpreter that runs the DeepSeek OCR runner.
- `GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT`: override path to the OCR runner script (defaults to `src/glossapi/ocr/deepseek/run_pdf_ocr_transformers.py`).
- `GLOSSAPI_DEEPSEEK_MODEL_DIR`: path to the downloaded `DeepSeek-OCR-2` snapshot.
- `GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH`: prepend extra library search paths when launching the OCR runner.

Standard OCR defaults:

- `runtime_backend='vllm'`
- `ocr_profile='markdown_grounded'`
- `max_new_tokens=2048`
- `repair_mode='auto'`
- `scheduler='auto'`
- `target_batch_pages=160`

The DeepSeek runners now default to `max_new_tokens=2048`. Do not leave the token cap implicit in one environment and explicit in another when comparing benchmarks.

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

## Triage & Parquet

- Triage always writes both:
  - Sidecar summaries: `sidecars/triage/{stem}.json` (per document)
  - Parquet updates: `download_results/download_results.parquet` (adds/updates rows)
- Default recommendation policy: enrich if `formula_total > 0` (skip only no‑math docs).
- Legacy heuristic (p90/pages thresholds) can be enabled with `GLOSSAPI_TRIAGE_HEURISTIC=1`.
