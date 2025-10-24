# GlossAPI Documentation

Welcome to the refreshed docs for GlossAPI, the GFOSS pipeline for turning academic PDFs into clean Markdown and metadata artifacts.

## Start here
- [Onboarding Guide](getting_started.md) — prerequisites, install choices, and first run.
- [Quickstart Recipes](quickstart.md) — common extraction/OCR flows in copy-paste form.
- [Lightweight PDF Corpus](lightweight_corpus.md) — 20 one-page PDFs for smoke testing without Docling or GPUs.

## Learn the pipeline
- [Pipeline Overview](pipeline.md) explains each stage and the emitted artifacts.
- [OCR & Math Enrichment](ocr_and_math_enhancement.md) covers Docling + RapidOCR usage.
- [Multi-GPU & Benchmarking](multi_gpu.md) shares scaling and scheduling tips.

## Configure and debug
- [Configuration](configuration.md) lists all environment knobs.
- [Troubleshooting](troubleshooting.md) captures the most common pitfalls.
- [AWS Job Distribution](aws_job_distribution.md) describes large-scale scheduling.

## Reference
- [Corpus API](api/corpus.md) details public methods and parameters.
- `docs/divio/` contains placeholder pages for the upcoming Divio restructuring—feel free to open PRs fleshing them out.
