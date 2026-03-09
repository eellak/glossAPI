# GlossAPI Documentation

Welcome to the refreshed docs for GlossAPI, the GFOSS pipeline for turning academic PDFs into clean Markdown and metadata artifacts.

## Start here
- [Onboarding Guide](getting_started.md) — prerequisites, install choices, and first run.
- [Quickstart Recipes](quickstart.md) — common extraction/OCR flows in copy-paste form.
- [Lightweight PDF Corpus](lightweight_corpus.md) — 20 one-page PDFs for smoke testing without Docling or GPUs.

## Understand the architecture
- [Architecture Overview](architecture/index.md) — the end-to-end staged model and why it exists.
- [Core Design Principles](architecture/core_design_principles.md) — the design constraints that shape the pipeline.
- [Docling Throughput and Batching](architecture/docling_throughput_and_batching.md) — how throughput and stability trade off.
- [Failure Recovery and Skiplist](architecture/docling_failure_recovery_and_skiplist.md) — how the pipeline survives problematic PDFs.
- [Greek Text Validation](architecture/greek_text_validation.md) — why extraction success is not enough for Greek corpora.
- [Metadata, Artifacts, and Run Diagnostics](architecture/metadata_artifacts_and_run_diagnostics.md) — how provenance and operational state are retained.
- [Artifact Layout and Stage Handoffs](architecture/artifact_layout_and_stage_handoffs.md) — how folders, filenames, and metadata glue the stages together.
- [Resumability, Recovery, and Retention](architecture/resumability_recovery_and_retention.md) — how the current design supports reruns and where storage pressure appears.
- [DeepSeek-Only Upgrade Roadmap](architecture/deepseek_only_upgrade_roadmap.md) — the staged simplification plan for OCR and dependency upgrades.

## Learn the pipeline
- [Pipeline Overview](pipeline.md) explains each stage and the emitted artifacts.
- [OCR & Math Enrichment](ocr_and_math_enhancement.md) covers DeepSeek OCR remediation and Docling-based enrichment.
- [Multi-GPU & Benchmarking](multi_gpu.md) shares scaling and scheduling tips.
- [Stage Reference](stages/index.md) breaks down each pipeline stage as a contract.

## Configure and debug
- [Configuration](configuration.md) lists all environment knobs.
- [Troubleshooting](troubleshooting.md) captures the most common pitfalls.
- [AWS Job Distribution](aws_job_distribution.md) describes large-scale scheduling.
- [Compatibility And Regression Matrix](testing/compatibility_matrix.md) defines the release-validation gates for the migration and upgrades.

## Reference
- [Corpus API](api/corpus.md) details public methods and parameters.
- `docs/divio/` contains placeholder pages for the upcoming Divio restructuring—feel free to open PRs fleshing them out.
