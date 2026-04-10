# GlossAPI Documentation

Welcome to the refreshed docs for GlossAPI, the GFOSS pipeline for turning academic PDFs into clean Markdown and metadata artifacts.

## Start here
- [Onboarding Guide](getting_started.md) — prerequisites, install choices, and first run.
- [Quickstart Recipes](quickstart.md) — common extraction/OCR flows in copy-paste form.
- [Lightweight PDF Corpus](lightweight_corpus.md) — 20 one-page PDFs for smoke testing without Docling or GPUs.

## Learn the pipeline
- [Code Map](code_map.md) links the main documentation ideas to the classes and files that implement them.
- [Pipeline Overview](pipeline.md) explains each stage and the emitted artifacts.
- [OCR & Math Enrichment](ocr_and_math_enhancement.md) covers DeepSeek OCR remediation and Docling-based enrichment.
- [OCR Repetition Policy](ocr_repetition_policy.md) pins the default repetition thresholds for word and LaTeX cleaning.
- [OCR Cleaning Runtime](architecture/ocr_cleaning_runtime.md) explains the shared clean/debug analyzer, ordering, and why the cleaner separates tables, numeric, LaTeX, hybrid, and text ownership.
- [Multi-GPU & Benchmarking](multi_gpu.md) shares scaling and scheduling tips.

## Configure and debug
- [Configuration](configuration.md) lists all environment knobs.
- [Troubleshooting](troubleshooting.md) captures the most common pitfalls.
- [AWS Job Distribution](aws_job_distribution.md) describes large-scale scheduling.

## Reference
- [Corpus API](api/corpus.md) gives the compact contract view of the main public methods.
- [Legacy Corpus API Notes](api_corpus_tmp.md) remains available while the docs are being consolidated.
