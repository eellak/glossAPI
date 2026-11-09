# Code Map

This page maps the main documentation ideas to the code that implements them. It is
meant to help you move from "what does GlossAPI do?" to "where do I change it?"
without reading the entire repo.

## Top-Level Entry Points

| Area | Main code | Responsibility |
| --- | --- | --- |
| Public package entry | `src/glossapi/__init__.py` | Lazy-exports `Corpus`, `GlossSectionClassifier`, `GlossDownloader`, and related classes without pulling heavy runtime dependencies at import time. |
| High-level orchestration | `src/glossapi/corpus/corpus_orchestrator.py` | Coordinates the end-to-end pipeline and owns the main folder/artifact conventions. |
| Phase-1 extraction engine | `src/glossapi/gloss_extract.py` | Defaults to PDFium with Rust layout labels; handles optional Docling selection, batching, timeouts, resumption and artifact export. |
| Streaming run and CLI | `src/glossapi/stages/run.py`, `src/glossapi/stages/scheduler.py`, `src/glossapi/cli.py` | `Corpus.run()` / `glossapi run`: plans the per-document DAG, runs stage workers, executes the corpus barriers (`stages/barriers.py`), retries interrupted attempts and publishes the products. |
| Provenance ledger | `src/glossapi/ledger/` | The corpus SQLite ledger `corpus.sqlite3`: schema and storage versions (`schema.py`, `compact.py`), the single writer and durability barriers (`writer.py`), the read side of `status`/`report` (`reader.py`) and the legacy import (`importer.py`). |
| Retention and disk budget | `src/glossapi/stages/retention.py`, `src/glossapi/stages/disk_budget.py`, `src/glossapi/stages/final_products.py` | `debug`/`memory_saving` deletion rules, the extreme-mode disk guard and the publication disk charge. |

## Pipeline Stages

| Stage | Main methods/classes | Notes |
| --- | --- | --- |
| Download | `Corpus.download()`, `GlossDownloader.download_files()` | Supports URL expansion, deduplication, checkpoints, per-domain scheduling, and resume. |
| Extract | `Corpus.prime_extractor()`, `Corpus.extract()`, `GlossExtract.ensure_extractor()`, `GlossExtract.extract_path()` | Handles backend choice, GPU preflight, and single- vs multi-GPU dispatch. |
| Clean / quality gate | `Corpus.clean()` | Runs the Rust cleaner and merges quality metrics back into parquet metadata. |
| OCR retry / math follow-up | `Corpus.ocr()`, `Corpus.formula_enrich_from_json()` | OCRs the pages page quality routes (default `page_routing="all"`) or only cleaner-flagged documents (`"flagged"`), and optionally performs Phase-2 math/code enrichment from JSON. |
| Sectioning | `Corpus.section()`, `GlossSection.to_parquet()` | Converts markdown documents into section rows for later classification. |
| Classification / annotation | `Corpus.annotate()`, `GlossSectionClassifier.classify_sections()`, `GlossSectionClassifier.fully_annotate()` | Scores sections with the packaged linear classifier (`section_model.py`, numpy only) and post-processes section labels into final document structure. |
| Export / triage | `Corpus.jsonl()`, `Corpus.triage_math()` | Produces training/export JSONL and computes routing hints for math-dense documents. |

## Backend and Runtime Helpers

| File | Responsibility |
| --- | --- |
| `src/glossapi/ocr/docling/pipeline.py` | Canonical builder for the layout-only Docling Phase-1 pipeline, including runtime tuning knobs for the current Docling API. |
| `src/glossapi/ocr/docling_pipeline.py` | Compatibility re-export for the canonical Docling pipeline builder. |
| `src/glossapi/ocr/utils/json_io.py` | Writes and reads compressed Docling JSON artifacts. |
| `src/glossapi/corpus/phase_ocr_math.py` | Runs external Docling plugin OCR remediation, math/code enrichment, and parquet status updates. |
| `src/glossapi/pdf_repair/tounicode.py`, `src/glossapi/scripts/pdf_font_repair.py` | Analyze embedded font Unicode maps, append conservative ToUnicode repairs, and report bounded CPU batches. |
| `src/glossapi/pdf_repair/extract.py` | Worker-local, bounded ToUnicode extract pre-pass; repairs from the source on every call (no reuse of `repaired_pdfs/` by modification time) and records font-repair provenance. |
| `src/glossapi/metrics.py` | Computes per-page parse/OCR/formula metrics from Docling conversions. |

## Rust Extensions

| Crate | Path | Purpose |
| --- | --- | --- |
| Cleaner | `rust/glossapi_rs_cleaner` | Markdown cleaning, script/noise filtering, and report generation used by `Corpus.clean()`. |
| Noise metrics | `rust/glossapi_rs_noise` | Fast quality metrics used by the broader pipeline and package build configuration. |
| PDF font repair | `rust/glossapi_rs_pdfrepair` | Native ToUnicode analysis and repair used by the extraction pre-pass (see its README). |
| Shared helpers | `rust/glossapi_rs_common` | Script metrics (Greek/Latin character and word counts) shared by the cleaner and noise crates. |

## Tests To Read First

| Test | Why it matters |
| --- | --- |
| `tests/test_corpus_phases_lightweight.py` | Extract, clean and section on the two lightweight PDFs; runs without GPU or Docling. |
| `tests/test_pipeline_smoke.py` | High-level example of the artifact flow through extract -> clean -> OCR -> section; needs torch, Docling and a CUDA GPU (otherwise it skips). |
| `tests/test_corpus_guards.py` | Shows the contract around backend selection and GPU preflight. |
| `tests/test_jsonl_export.py` | Shows how final JSONL export merges cleaned markdown, parquet metadata, and math metrics. |
| `tests/test_ocr_plugin_dispatch.py` | Covers the external OCR dispatch contract and backend validation. |

## If You Need To Change...

- Download scheduling or resume behavior: start in `src/glossapi/gloss_downloader.py`.
- Phase-1 parsing, worker fanout, or artifact generation: start in `src/glossapi/corpus/phase_extract.py`, `src/glossapi/corpus/corpus_orchestrator.py`, and `src/glossapi/gloss_extract.py`.
- Docling pipeline wiring or runtime tuning: start in `src/glossapi/ocr/docling/pipeline.py` and `src/glossapi/gloss_extract.py`.
- Section labels or section-annotation rules: start in `src/glossapi/gloss_section_classifier.py`.
- Output folder contracts or stage sequencing: start in `src/glossapi/corpus/corpus_orchestrator.py`.
