# GlossAPI

GlossAPI is a GPU-ready document processing pipeline from [GFOSS](https://gfoss.eu/) that turns academic PDFs into structured Markdown, cleans noisy text with Rust extensions, and optionally enriches math/code content.

## Why GlossAPI
- Handles download → extraction → cleaning → sectioning in one pipeline.
- Ships safe PyPDFium extraction plus Docling for structured extraction and DeepSeek-OCR-2 for OCR remediation.
- Rust-powered cleaner/noise metrics keep Markdown quality predictable.
- Greek-first metadata and section classification tuned for academic corpora.
- Modular Corpus API lets you resume from any stage or plug into existing flows.

## Quickstart (local repo)

```bash
git clone https://github.com/eellak/glossAPI.git
cd glossAPI
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run the lightweight PDF corpus (no GPU/Docling required)
python - <<'PY'
from pathlib import Path
from glossapi import Corpus

input_dir = Path("samples/lightweight_pdf_corpus/pdfs")
output_dir = Path("artifacts/lightweight_pdf_run")
output_dir.mkdir(parents=True, exist_ok=True)

corpus = Corpus(input_dir, output_dir)
corpus.extract(input_format="pdf")  # Safe PyPDFium backend by default
PY
```

- Compare the generated Markdown in `artifacts/lightweight_pdf_run/markdown/`
  with `samples/lightweight_pdf_corpus/expected_outputs.json` for a fast smoke check.
- Rebuild the corpus anytime with `python samples/lightweight_pdf_corpus/generate_pdfs.py`.

### Corpus usage contract
`Corpus` is the organizing surface: keep contributions wired through the phase methods (`download()`, `extract()`, `clean()`, `ocr()`, `section()`, `annotate()`, `export/jsonl*()`). The intended use is a short script chaining those calls; avoid bespoke monkeypatches or side channels so resumability and artifact layout stay consistent.

## Automated Environment Profiles

Use `dependency_setup/setup_glossapi.sh` for the Docling environment, or `dependency_setup/setup_deepseek_uv.sh` for the dedicated DeepSeek OCR runtime:

```bash
# Docling / main GlossAPI environment
./dependency_setup/setup_glossapi.sh --mode docling --venv dependency_setup/.venvs/docling --run-tests

# DeepSeek OCR runtime (uv-managed)
./dependency_setup/setup_deepseek_uv.sh \
  --venv dependency_setup/.venvs/deepseek \
  --model-root /path/to/deepseek-ocr-2-model \
  --download-model \
  --run-tests --smoke-test
```

`setup_glossapi.sh --mode deepseek` now delegates to the same uv-based installer. `setup_deepseek_uv.sh` uses `uv venv` + `uv sync`, installs the Rust extensions in editable mode, and can download `deepseek-ai/DeepSeek-OCR-2` with `huggingface_hub`.
The uv-managed DeepSeek runtime is OCR-only on purpose: it installs `glossapi[deepseek]` and does not carry the Docling layout stack.

If you want a guided install that asks which phases you plan to use, run:

```bash
python install_glossapi.py
```

That wizard keeps browser-gated download support (`playwright`) and the dedicated DeepSeek OCR runtime out of the main environment unless you explicitly select them.

## Browser-Gated Download Mode

`Corpus.download(...)` now supports three high-level routes for file acquisition:

- `download_mode="standard"`: direct HTTP downloader only
- `download_mode="auto"`: direct HTTP first, then browser-assisted recovery when the response is a recoverable browser-gated interstitial
- `download_mode="browser"`: go straight to browser-assisted acquisition for known browser-gated file endpoints

Use `browser_mode=True` as a legacy alias for `download_mode="browser"`.

### Policy-driven routing

If you know which domains require browser bootstrap, route them with a policy file instead of probing every URL:

```yaml
default:
  downloader: standard

rules:
  - match:
      domains: [eur-lex.europa.eu]
    downloader: browser

  - match:
      url_regex: "https://example.org/protected/.*"
    downloader: auto
```

```python
from glossapi import Corpus

corpus = Corpus(input_dir="out", output_dir="out")
corpus.download(
    input_parquet="input_urls.parquet",
    download_policy_file="download_policy.yml",
)
```

### Operational notes

- Browser mode is for browser-gated file endpoints, not viewer-only sources.
- Browser sessions are cached per domain so a successful bootstrap can be reused across multiple files.
- Successful downloads still land in `downloads/`; extraction continues to consume only real files from that directory.
- Viewer-style sources still fail cleanly in `download_results/*.parquet` and do not create fake files.

### Regression strategy

The checked-in browser download tests use mocked browser/session flows and fake PDF bytes rather than hard-coded live URLs.

For manual smoke checks against live browser-gated sources, build an ad hoc parquet locally and run it outside the committed test suite.

**DeepSeek runtime checklist**
- Run `python -m glossapi.ocr.deepseek.preflight` from the DeepSeek venv to fail fast before OCR.
- Export these to force the real runtime and avoid silent stub output:
  - `GLOSSAPI_DEEPSEEK_ALLOW_CLI=1`
  - `GLOSSAPI_DEEPSEEK_ALLOW_STUB=0`
  - `GLOSSAPI_DEEPSEEK_PYTHON=/path/to/deepseek/venv/bin/python`
  - `GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT=/path/to/glossAPI/src/glossapi/ocr/deepseek/run_pdf_ocr_transformers.py`
  - `GLOSSAPI_DEEPSEEK_MODEL_DIR=/path/to/deepseek-ocr-2-model/DeepSeek-OCR-2`
- The default fallback locations already point at the in-repo Transformers runner and `${REPO_ROOT}/deepseek-ocr-2-model/DeepSeek-OCR-2`.
- `flash-attn` is optional. The runner uses `flash_attention_2` when available and falls back to `eager` otherwise.

## Choose Your Install Path

| Scenario | Commands | Notes |
| --- | --- | --- |
| Pip users | `pip install glossapi` | Fast vanilla evaluation with minimal dependencies. |
| Docling environment | `./dependency_setup/setup_glossapi.sh --mode docling` | Creates the main GlossAPI venv for extraction, cleaning, sectioning, and enrichment. |
| DeepSeek environment | `./dependency_setup/setup_deepseek_uv.sh` | Creates a separate uv-managed OCR runtime pinned to the tested Transformers/Torch stack. |
| Manual editable install | `pip install -e .` after cloning | Keep this if you prefer to manage dependencies by hand. |
| Conda-based stacks | `scripts/setup_conda.sh` | Provisions Python 3.10 env + Rust + editable install for Amazon Linux/SageMaker. |

See the refreshed docs (`docs/index.md`) for detailed environment notes, CUDA/ORT combinations, and troubleshooting tips.

## Repo Landmarks
- `docs/code_map.md`: fast map from pipeline ideas to implementing classes and files.
- `docs/pipeline.md`: stage contracts, key parameters, and artifact outputs.
- `samples/lightweight_pdf_corpus/`: 20 one-page PDFs with manifest + expected Markdown.
- `src/glossapi/`: Corpus pipeline, cleaners, and orchestration logic.
- `tests/test_pipeline_smoke.py`: Minimal regression entry point (uses the lightweight corpus).
- `docs/`: MkDocs site with onboarding, pipeline recipes, and configuration guides.

## Pipeline map

Use this as the shortest path from a documentation concept to the public call that implements it.

| Stage | Main call | Important parameters | Writes |
| --- | --- | --- | --- |
| Download | `Corpus.download(...)` | `input_parquet`, `links_column`, `parallelize_by`, `download_mode="standard"|"auto"|"browser"`, `download_policy_file`, downloader kwargs | `downloads/`, `download_results/*.parquet` |
| Extract (Phase-1) | `Corpus.extract(...)` | `input_format`, `phase1_backend`, `use_gpus`, `workers_per_device`, `export_doc_json`, `emit_formula_index` | `markdown/<stem>.md`, `json/<stem>.docling.json(.zst)`, `json/metrics/*.json` |
| Clean | `Corpus.clean(...)` | `threshold`, `drop_bad`, `empty_char_threshold`, `empty_min_pages` | `clean_markdown/<stem>.md`, updated parquet metrics/flags |
| OCR / math follow-up | `Corpus.ocr(...)` | `mode`, `fix_bad`, `math_enhance`, `use_gpus`, `devices` | refreshed `markdown/<stem>.md`, optional `json/<stem>.latex_map.jsonl` |
| Section | `Corpus.section()` | uses cleaner/parquet outputs to choose inputs | `sections/sections_for_annotation.parquet` |
| Annotate | `Corpus.annotate(...)` | `annotation_type`, `fully_annotate` | `classified_sections.parquet`, `fully_annotated_sections.parquet` |
| Triage math density | `Corpus.triage_math()` | no required args | updated `download_results/*.parquet` routing columns |
| JSONL export | `Corpus.jsonl(...)` | `output_path` | merged training/export JSONL |

## Contributing
- Run `pytest tests/test_pipeline_smoke.py` for a fast end-to-end check.
- Regenerate the lightweight corpus via `generate_pdfs.py` and commit the updated PDFs + manifest together.
- Prefer `uv` or `pip` editable installs so Rust extensions rebuild locally.

Open an issue or PR if you spot drift between expected outputs and the pipeline, or if you have doc updates for the new Divio skeleton.

## License

This project is licensed under the [EUPL 1.2](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12).
