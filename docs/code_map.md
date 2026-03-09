# Code Map

This page maps the main documentation ideas to the code that implements them. It is
meant to help you move from "what does GlossAPI do?" to "where do I change it?"
without reading the entire repo.

## Top-Level Entry Points

| Area | Main code | Responsibility |
| --- | --- | --- |
| Public package entry | `src/glossapi/__init__.py` | Applies the RapidOCR patch on import and exports `Corpus`, `GlossSectionClassifier`, `GlossDownloader`, and related classes. |
| High-level orchestration | `src/glossapi/corpus.py` | Coordinates the end-to-end pipeline and owns the main folder/artifact conventions. |
| Phase-1 extraction engine | `src/glossapi/gloss_extract.py` | Builds/reuses Docling converters, handles safe vs Docling backend selection, batching, timeouts, resumption, and artifact export. |

## Pipeline Stages

| Stage | Main methods/classes | Notes |
| --- | --- | --- |
| Download | `Corpus.download()`, `GlossDownloader.download_files()` | Supports URL expansion, deduplication, checkpoints, per-domain scheduling, and resume. |
| Extract | `Corpus.prime_extractor()`, `Corpus.extract()`, `GlossExtract.ensure_extractor()`, `GlossExtract.extract_path()` | Handles backend choice, GPU preflight, and single- vs multi-GPU dispatch. |
| Clean / quality gate | `Corpus.clean()` | Runs the Rust cleaner and merges quality metrics back into parquet metadata. |
| OCR retry / math follow-up | `Corpus.ocr()`, `Corpus.formula_enrich_from_json()` | Re-runs OCR only for flagged documents and optionally performs Phase-2 math/code enrichment from JSON. |
| Sectioning | `Corpus.section()`, `GlossSection.to_parquet()` | Converts markdown documents into section rows for later classification. |
| Classification / annotation | `Corpus.annotate()`, `GlossSectionClassifier.classify_sections()`, `GlossSectionClassifier.fully_annotate()` | Runs the SVM classifier and post-processes section labels into final document structure. |
| Export / triage | `Corpus.jsonl()`, `Corpus.triage_math()` | Produces training/export JSONL and computes routing hints for math-dense documents. |

## Backend and Runtime Helpers

| File | Responsibility |
| --- | --- |
| `src/glossapi/_pipeline.py` | Canonical builders for layout-only and RapidOCR-backed Docling pipelines. |
| `src/glossapi/rapidocr_safe.py` | Monkey-patch/shim for Docling 2.48.x so problematic OCR crops do not crash whole documents. |
| `src/glossapi/_rapidocr_paths.py` | Resolves packaged RapidOCR ONNX models and Greek keys, with env-var override support. |
| `src/glossapi/ocr_pool.py` | Reuses RapidOCR model instances where possible. |
| `src/glossapi/json_io.py` | Writes and reads compressed Docling JSON artifacts. |
| `src/glossapi/triage.py` | Summarizes per-page formula density and updates parquet routing metadata. |
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
| `tests/test_rapidocr_patch.py` | Covers the Docling/RapidOCR compatibility patch and fallback paths. |

## If You Need To Change...

- Download scheduling or resume behavior: start in `src/glossapi/gloss_downloader.py`.
- Phase-1 parsing, OCR selection, or artifact generation: start in `src/glossapi/corpus.py` and `src/glossapi/gloss_extract.py`.
- Docling/RapidOCR wiring or provider issues: start in `src/glossapi/_pipeline.py`, `src/glossapi/rapidocr_safe.py`, and `src/glossapi/_rapidocr_paths.py`.
- Section labels or section-annotation rules: start in `src/glossapi/gloss_section_classifier.py`.
- Output folder contracts or stage sequencing: start in `src/glossapi/corpus.py`.
