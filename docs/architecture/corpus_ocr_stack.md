# Corpus OCR Stack

This note documents the boundary between corpus orchestration and OCR runtime
code after the `corpus/ocr/` extraction.

## Design Boundary

There are now two layers:

- `src/glossapi/corpus/ocr/`
  - owns corpus-facing policy
  - decides which documents should run OCR or math
  - updates parquet metadata and stage-owned artifacts
  - keeps `Corpus.ocr()` and `Corpus.formula_enrich_from_json()` readable
- `src/glossapi/ocr/deepseek/`
  - owns DeepSeek runtime execution
  - builds worker commands and environments
  - runs single- and multi-worker OCR
  - handles runtime telemetry, queue state, and shard reassembly

That split is intentional. Corpus code should not need to know how vLLM workers
are launched, and runtime code should not need to know parquet policy.

## Corpus OCR Package Map

The corpus package is organized by responsibility instead of one large phase
file:

- `config.py`
  - normalizes the public `Corpus.ocr()` request into an internal `OcrRequest`
  - is the single place where shared DeepSeek defaults are resolved
- `targets.py`
  - loads OCR candidate rows from parquet
  - applies skip-list and completed-work filtering
  - collapses chunk-like rows back to canonical source PDFs
- `dispatch.py`
  - forwards a normalized OCR request into the DeepSeek runtime
- `artifacts.py`
  - persists OCR success back into parquet
  - owns the direct OCR artifact fields (`text`, markdown relpath, metrics relpath, sha256)
- `math_targets.py`
  - discovers Docling JSON stems and filters math follow-up candidates
- `math_pipeline.py`
  - runs single-process Phase-2 math enrichment from Docling JSON
  - updates math results in parquet
- `math_worker.py`
  - contains the spawned multi-GPU math worker entrypoint
- `math_runtime.py`
  - supervises multi-worker math execution and skip-list updates
- `pipeline.py`
  - coordinates mode selection (`ocr_bad`, `math_only`, `ocr_bad_then_math`)
  - is the only corpus-side module that spans OCR and math in one control flow
- `phase_ocr_math.py`
  - remains a thin compatibility layer for the public `Corpus` mixin methods

## Call Flow

For OCR remediation:

1. `Corpus.ocr()` calls `normalize_ocr_request()`.
2. `pipeline.run_ocr_phase()` loads OCR selection through `targets.build_ocr_selection()`.
3. `dispatch.run_deepseek_ocr()` hands work to `glossapi.ocr.deepseek.runner`.
4. `artifacts.persist_ocr_success()` writes canonical OCR results back into parquet.
5. `refresh_cleaner_after_ocr()` reruns the cleaner so downstream metrics stay in sync.

For math-only follow-up:

1. `pipeline.run_ocr_phase()` selects stems through `math_targets.py`.
2. `math_runtime.run_math_enrichment()` decides between single-process and multi-GPU execution.
3. `math_pipeline.formula_enrich_from_json()` performs per-stem JSON-driven enrichment.

## Safe Change Rules

If you need to change corpus OCR behavior, prefer these rules:

- Change selection policy in `targets.py`, not in the runtime runner.
- Change OCR result parquet fields in `artifacts.py`, not in `pipeline.py`.
- Change public defaults in `config.py` and `glossapi/ocr/deepseek/defaults.py` together.
- Keep `phase_ocr_math.py` thin. It should translate the public API to package helpers, not grow new policy.
- Keep runtime-specific behavior in `glossapi/ocr/deepseek/`, even when a corpus call path triggers it.

## Tests That Protect This Boundary

- `tests/test_ocr_dispatch_backends.py`
- `tests/test_ocr_backends_smoke.py`
- `tests/test_math_policy.py`
- `tests/test_corpus_ocr_modules.py`
- `tests/test_deepseek_runner_contract.py`
