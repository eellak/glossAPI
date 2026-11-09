# GlossAPI Documentation

Welcome to the refreshed docs for GlossAPI, the GFOSS pipeline for turning academic PDFs into clean Markdown and metadata artifacts.

## Greek OCR: code, models and data

**The Greek OCR toolkit is the only supported OCR engine in GlossAPI 0.2.0.**

- **[OCR toolkit on GitHub](https://github.com/fffoivos/greek-ocr-toolkit)** — source code for the OCR integration.
- **[Greek OCR models on Hugging Face](https://huggingface.co/fffoivos/greek-ocr)** — the repository used by GlossAPI's default x1 recognizer.
- **[Greek OCR Lines dataset on Hugging Face](https://huggingface.co/datasets/fffoivos/greek-ocr-lines)** — the toolkit's upstream line-data project.

GlossAPI pins the toolkit source and recognition model revision. It routes selected physical pages to OCR and preserves retained native text. Read [OCR setup, pins and supported inputs](ocr_plugin_migration.md) and the [OCR stage contract](stages/ocr.md).

Core text-layer extraction produces layout-aware Markdown with PDFium and Rust; it needs neither Docling nor a GPU. `GLOSSAPI_MARKDOWN_LAYOUT=plain` selects plain text. Install the OCR package for recognition, or the optional layout package for Docling layout and math/code enrichment. See [installation](getting_started.md).

## Start here
- [Onboarding Guide](getting_started.md) — prerequisites, install choices, and first run.
- [Quickstart Recipes](quickstart.md) — common extraction/OCR flows in copy-paste form.
- [Lightweight PDF Corpus](lightweight_corpus.md) — two one-page PDFs for smoke testing without Docling or GPUs.

## Learn the pipeline
- [Code Map](code_map.md) links the main documentation ideas to the classes and files that implement them.
- [Pipeline Overview](pipeline.md) explains each stage and the emitted artifacts.
- [OCR & Math Enrichment](ocr_and_math_enhancement.md) covers Greek OCR toolkit remediation and optional Docling-based enrichment.
- [OCR Repetition Policy](ocr_repetition_policy.md) pins the default repetition thresholds for word and LaTeX cleaning.
- [OCR Cleaning Runtime](architecture/ocr_cleaning_runtime.md) explains the shared clean/debug analyzer, ordering, and why the cleaner separates tables, numeric, LaTeX, hybrid, and text ownership.
- [Multi-GPU & Benchmarking](multi_gpu.md) shares scaling and scheduling tips.

## Configure and debug
- [Configuration](configuration.md) lists the main environment knobs; the OCR toolkit and cache variables are in
  [OCR setup](ocr_plugin_migration.md).
- [Troubleshooting](troubleshooting.md) captures the most common pitfalls.
- [Corpus-wide barriers](architecture/corpus_barriers.md) lists the points where a streaming run waits for the whole
  corpus.

## Reference
- [Corpus API](api/corpus.md) gives the compact contract view of the main public methods.
- [CLI JSON schema](cli_json_schema.md) describes the `glossapi` commands, their JSON output, exit codes and typed errors.
- [LLM guide](llm_guide.md) is the guide `glossapi guide` prints, rendered as Markdown.
- [Resumability, Recovery, and Retention](architecture/resumability_recovery_and_retention.md) explains resume, retries, retention settings and durability of a streaming run.
