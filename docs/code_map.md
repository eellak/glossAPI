# Code Map

This page maps the main documentation ideas to the code that implements them. It is
meant to help you move from "what does GlossAPI do?" to "where do I change it?"
without reading the entire repo.

## Top-Level Entry Points

| Area | Main code | Responsibility |
| --- | --- | --- |
| Public package entry | `src/glossapi/__init__.py` | Lazy-exports `Corpus`, `GlossSectionClassifier`, `GlossDownloader`, and related classes without pulling heavy runtime dependencies at import time. |
| High-level orchestration | `src/glossapi/corpus/corpus_orchestrator.py` | Coordinates the end-to-end pipeline and owns the main folder/artifact conventions. |
| Phase-1 extraction engine | `src/glossapi/gloss_extract.py` | Builds/reuses Docling converters, handles safe vs Docling backend selection, batching, timeouts, resumption, and artifact export. |

## Pipeline Stages

| Stage | Main methods/classes | Notes |
| --- | --- | --- |
| Download | `Corpus.download()`, `GlossDownloader.download_files()` | Supports URL expansion, deduplication, checkpoints, per-domain scheduling, and resume. |
| Extract | `Corpus.prime_extractor()`, `Corpus.extract()`, `GlossExtract.ensure_extractor()`, `GlossExtract.extract_path()` | Handles backend choice, GPU preflight, and single- vs multi-GPU dispatch. |
| Clean / quality gate | `Corpus.clean()` | Runs the Rust cleaner and merges quality metrics back into parquet metadata. |
| OCR retry / math follow-up | `Corpus.ocr()`, `Corpus.formula_enrich_from_json()`, `src/glossapi/corpus/ocr/` | Re-runs OCR only for flagged documents, updates parquet state, and optionally performs Phase-2 math/code enrichment from JSON. |
| Sectioning | `Corpus.section()`, `GlossSection.to_parquet()` | Converts markdown documents into section rows for later classification. |
| Classification / annotation | `Corpus.annotate()`, `GlossSectionClassifier.classify_sections()`, `GlossSectionClassifier.fully_annotate()` | Runs the SVM classifier and post-processes section labels into final document structure. |
| Export / triage | `Corpus.jsonl()`, `Corpus.triage_math()` | Produces training/export JSONL and computes routing hints for math-dense documents. |

## Backend and Runtime Helpers

| File | Responsibility |
| --- | --- |
| `src/glossapi/ocr/docling/pipeline.py` | Canonical builder for the layout-only Docling Phase-1 pipeline, including runtime tuning knobs for the current Docling API. |
| `src/glossapi/ocr/docling_pipeline.py` | Compatibility re-export for the canonical Docling pipeline builder. |
| `src/glossapi/ocr/deepseek/defaults.py` | Shared DeepSeek OCR defaults used by the Corpus API, lane runner, and CLI entrypoints. |
| `src/glossapi/ocr/deepseek/launcher.py` | Owns subprocess command/env construction for DeepSeek OCR workers, including runtime library-path setup. |
| `src/glossapi/corpus/ocr/config.py` | Normalizes the public `Corpus.ocr()` request into an internal `OcrRequest`, including shared DeepSeek defaults and legacy flag compatibility. |
| `src/glossapi/corpus/ocr/targets.py` | Selects OCR candidates from parquet, skip-lists, and existing completion signals, then maps chunk rows back to canonical PDFs. |
| `src/glossapi/corpus/ocr/artifacts.py` | Persists OCR success into parquet and owns the canonical OCR artifact fields copied back into metadata. |
| `src/glossapi/corpus/ocr/pipeline.py` | Corpus-side coordinator for OCR-only, math-only, and OCR-then-math modes. |
| `src/glossapi/corpus/ocr/math_pipeline.py` | Single-process Phase-2 math enrichment from Docling JSON and parquet result updates. |
| `src/glossapi/corpus/ocr/math_runtime.py` | Multi-GPU math orchestration, worker supervision, and skip-list integration. |
| `src/glossapi/ocr/deepseek/runner.py` | Public DeepSeek OCR shim used by `Corpus.ocr()`, keeping the stable orchestration entrypoint and monkeypatch-sensitive helpers. |
| `src/glossapi/ocr/deepseek/runner_planning.py` | Pure lane/batch planning, device resolution, and automatic vLLM batch-size helpers used by the runner shim. |
| `src/glossapi/ocr/deepseek/runner_runtime_support.py` | Runtime summaries, GPU preflight, telemetry, XID capture, and worker process lifecycle helpers used by the runner shim. |
| `src/glossapi/ocr/deepseek/runner_reassembly.py` | Reassembles exact-fill shard outputs back into canonical markdown/metrics artifacts. |
| `src/glossapi/ocr/deepseek/runtime_paths.py` | Resolves which Python interpreter should be used for the DeepSeek runtime. |
| `src/glossapi/ocr/deepseek/scheduling.py` | Builds whole-document, fixed-shard, and exact-fill OCR work plans. |
| `src/glossapi/ocr/deepseek/work_queue.py` | Persists worker queue state, retries, and repair batches for vLLM OCR runs. |
| `src/glossapi/ocr/utils/json_io.py` | Writes and reads compressed Docling JSON artifacts. |
| `src/glossapi/corpus/phase_ocr_math.py` | Thin compatibility layer that keeps the public `Corpus` OCR/mathematics methods stable while delegating to `src/glossapi/corpus/ocr/`. |
| `src/glossapi/metrics.py` | Computes per-page parse/OCR/formula metrics from Docling conversions. |

## Rust Extensions

| Crate | Path | Purpose |
| --- | --- | --- |
| Cleaner | `rust/glossapi_rs_cleaner` | Markdown cleaning, script/noise filtering, and report generation used by `Corpus.clean()`. |
| Noise metrics | `rust/glossapi_rs_noise` | Fast quality metrics used by the broader pipeline and package build configuration. |

## Tests To Read First

| Test | Why it matters |
| --- | --- |
| `tests/test_pipeline_smoke.py` | Best high-level example of the intended artifact flow through extract -> clean -> OCR -> section. |
| `tests/test_corpus_guards.py` | Shows the contract around backend selection and GPU preflight. |
| `tests/test_jsonl_export.py` | Shows how final JSONL export merges cleaned markdown, parquet metadata, and math metrics. |
| `tests/test_ocr_dispatch_backends.py` | Covers the DeepSeek-only OCR dispatch contract and backend validation. |
| `tests/test_deepseek_runner_contract.py` | Covers DeepSeek launcher defaults, CLI wiring, and canonical output guarantees. |
| `tests/test_deepseek_multi_gpu_runtime.py` | Covers worker env setup, runtime-library discovery, and durable queue behavior. |

## If You Need To Change...

- Download scheduling or resume behavior: start in `src/glossapi/gloss_downloader.py`.
- Phase-1 parsing, worker fanout, or artifact generation: start in `src/glossapi/corpus/phase_extract.py`, `src/glossapi/corpus/corpus_orchestrator.py`, and `src/glossapi/gloss_extract.py`.
- Docling pipeline wiring or runtime tuning: start in `src/glossapi/ocr/docling/pipeline.py` and `src/glossapi/gloss_extract.py`.
- DeepSeek OCR defaults, launch envs, or benchmark wiring: start in `src/glossapi/ocr/deepseek/defaults.py`, `src/glossapi/ocr/deepseek/launcher.py`, `src/glossapi/ocr/deepseek/runner.py`, and `src/glossapi/scripts/openarchives_ocr_run_node.py`.
- DeepSeek runner internals, worker lifecycle, or shard reassembly: start in `src/glossapi/ocr/deepseek/runner.py` and then read `docs/architecture/deepseek_runner_stack.md`.
- Corpus OCR target selection, parquet updates, or Phase-2 math policy: start in `src/glossapi/corpus/ocr/` and read `docs/architecture/corpus_ocr_stack.md` first.
- Section labels or section-annotation rules: start in `src/glossapi/gloss_section_classifier.py`.
- Output folder contracts or stage sequencing: start in `src/glossapi/corpus/corpus_orchestrator.py`.
