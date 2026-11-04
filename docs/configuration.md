# Configuration & Environment Variables

This page lists the main knobs you can use to tune GlossAPI.

## GPU & Providers

- `CUDA_VISIBLE_DEVICES`: restrict/assign visible GPUs, e.g. `export CUDA_VISIBLE_DEVICES=0,1`.
- `GLOSSAPI_DOCLING_DEVICE`: preferred device for Docling (inside a worker), e.g. `cuda:0`.

## OCR & Parsing

- `GLOSSAPI_IMAGES_SCALE`: image scale hint for OCR/layout (default ~1.1–1.25).
- `GLOSSAPI_FORMULA_BATCH`: inline CodeFormula batch size (default `16`).
- `GLOSSAPI_MARKDOWN_LAYOUT`: the safe backend's Markdown, `layout_v2` (default: layout-aware Markdown from Rust layout
  v2, no Docling) or `plain` (the plain page text as before K5-A2). Part of the extract stage identity; see
  [Extract stage](stages/extract.md#layout-aware-markdown-layout-v2-core-install).

### Batch Policy & PDF Backend

GlossAPI exposes two Phase‑1 profiles. Use `Corpus.extract(..., phase1_backend='docling')` to switch from the default safe backend. The legacy environment variables `GLOSSAPI_BATCH_POLICY` and `GLOSSAPI_BATCH_MAX` are still parsed for backward compatibility but emit a deprecation warning and will be removed in a future release.

Regardless of backend, the extractor clamps OMP/OpenBLAS/MKL pools to one thread per worker so multi‑GPU runs do not explode thread counts.

### Docling Runtime Tuning

These optional knobs map directly to current Docling `PdfPipelineOptions` fields and are mainly useful for benchmarking on strong GPUs:

- `GLOSSAPI_DOCLING_MAX_BATCH_FILES`: override the number of PDF documents a single Phase‑1 Docling worker processes per extractor batch. Defaults to `1` in GlossAPI for stability; raise it deliberately when benchmarking fresh A100 nodes.
- `GLOSSAPI_DOCLING_BATCH_TARGET_PAGES`: target page budget for each queued multi‑GPU Docling work item. Defaults to `256`; lower it when a single worker hoards long PDFs, raise it when a strong GPU can keep larger mixed bundles resident.
- `GLOSSAPI_DOCLING_LAYOUT_BATCH_SIZE`: override Docling `layout_batch_size`.
- `GLOSSAPI_DOCLING_PAGE_BATCH_SIZE`: override Docling `settings.perf.page_batch_size` so Phase‑1 can raise or lower the number of pages each device keeps in flight internally without changing GlossAPI queue semantics.
- `GLOSSAPI_DOCLING_TABLE_BATCH_SIZE`: override Docling `table_batch_size`.
- `GLOSSAPI_DOCLING_OCR_BATCH_SIZE`: override Docling `ocr_batch_size` even though Phase‑1 OCR stays disabled.
- `GLOSSAPI_DOCLING_QUEUE_MAX_SIZE`: override Docling `queue_max_size`.
- `GLOSSAPI_DOCLING_DOCUMENT_TIMEOUT`: override Docling `document_timeout`.
- `GLOSSAPI_DOCLING_BATCH_POLL_INTERVAL`: override Docling `batch_polling_interval_seconds`.

### External OCR engines

Install `python install_glossapi.py ocr --device cpu` (or `mps` / `cuda126` /
`cuda132`). `Corpus.ocr(mode='ocr_bad')` defaults to the pinned Greek R8+x1
plugin; explicit `ocr_options` still select a configured external engine.
See [OCR plugin migration](ocr_plugin_migration.md).

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

## Runs, ledger and logs

- `GLOSSAPI_STAGE_PROFILE`: unset for the default `cutover` profile, which writes the ledger `corpus.sqlite3` and
  decides from it. `legacy` runs the pre-ledger stage code as a comparison control (its wrappers only record
  observations); `sequential-adapters` is the earlier qualification profile. See
  [Resumability, recovery and retention](architecture/resumability_recovery_and_retention.md).
- `GLOSSAPI_CONVERSION_LOG_FILE`: where the extractor writes its conversion log. Standalone scripts that construct the
  extractor write `conversion.log` in the working directory unless this names another path; stage workers of a
  streaming run write it inside their own work directory. `conversion.log` is ignored by git and is not part of the
  source tree; point the variable outside the checkout for long runs.

## Performance & Caches

- `OMP_NUM_THREADS` / `MKL_NUM_THREADS`: cap CPU threads to avoid oversubscription.
- Cache locations: `HF_HOME`, `XDG_CACHE_HOME`, `DOCLING_CACHE_DIR`.

## Worker Logging

- `GLOSSAPI_WORKER_LOG_DIR`: override the directory used for per-worker logs and `gpu<N>.current` markers (defaults to `logs/ocr_workers/` or `logs/math_workers/` under the output directory).
- `GLOSSAPI_WORKER_LOG_VERBOSE` = `1|0` (default `1`): emit (or suppress) the GPU binding banner each worker prints on startup.

## Test suite

- `GLOSSAPI_REFERENCE_REQUIRED=1`: a missing or different pinned reference oracle fails its test instead of skipping
  (set by `setup_glossapi.sh --run-tests`).
- `GLOSSAPI_OCR_GOLDEN_DIR`: root of the external real-page OCR golden fixture (`manifest.jsonl`, `inputs/`,
  `expected/`; build it with `glossapi.scripts.build_ocr_golden_pages`). `tests/test_ocr_golden_pages.py` skips
  without it and fails when it is set but holds no `manifest.jsonl`. Owner-private scans are never committed.
- `GLOSSAPI_NATIVE_WITNESS_PDF`: the `ABO_768.pdf` witness for the Docling native-cache test in
  `tests/test_batched_extract.py` (sha256 `a58e3a241c3af817a95892fc7d3db437ccaa616a7f57dbd5e24974b829eb0b05`). The
  test skips when the file is absent and fails when its bytes differ.
- `GLOSSAPI_REQUIRE_GPU=1`: the GPU-only tests in `tests/test_pipeline_smoke.py` fail instead of skipping when no CUDA
  GPU is visible.
- The exact font-repair rates in `tests/test_pdf_font_repair.py` are bound to the PDF, dictionary and PDFium library
  hashes. With other inputs the test still checks the repair invariants and skips only the exact rate, naming each
  hash that differs.
- Do not wrap pytest in an inherited `alarm` (for example `perl -e 'alarm N; exec pytest ...'`). The timer survives
  `exec`, font repair refuses to replace an active `SIGALRM` timer, and the repair tests then fail with status
  `error`. Bound the wait from outside instead (a background job and a watcher).

## Triage & Parquet

- Triage always writes both:
  - Sidecar summaries: `sidecars/triage/{stem}.json` (per document)
  - Parquet updates: `download_results/download_results.parquet` (adds/updates rows)
- Default recommendation policy: enrich if `formula_total > 0` (skip only no‑math docs).
- Legacy heuristic (p90/pages thresholds) can be enabled with `GLOSSAPI_TRIAGE_HEURISTIC=1`.
