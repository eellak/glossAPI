# GlossAPI

[![PyPI Status](https://img.shields.io/pypi/v/glossapi?logo=pypi)](https://pypi.org/project/glossapi/)

GlossAPI is a GPU-ready document processing pipeline from [GFOSS](https://gfoss.eu/) that turns academic PDFs into structured Markdown, cleans noisy text with Rust extensions, and optionally enriches math/code content.

## Why GlossAPI
- Handles download → extraction → cleaning → sectioning in one pipeline.
- Ships safe PyPDFium extraction plus Docling/RapidOCR for high-throughput OCR.
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

## Choose Your Install Path

| Scenario | Commands | Notes |
| --- | --- | --- |
| Pip users | `pip install glossapi` | Quick evaluation; GPU extras optional. |
| Local development | `pip install -e .` after cloning | Builds Rust extensions automatically. |
| Conda-based stacks | `scripts/setup_conda.sh` | Provisions Python 3.10 env + Rust + editable install. |
| GPU acceleration | `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1` | Optional Docling layout + math enrichment boost. |

See the refreshed docs (`docs/index.md`) for detailed environment notes, CUDA/ORT combinations, and troubleshooting tips.

## Repo Landmarks
- `samples/lightweight_pdf_corpus/`: 20 one-page PDFs with manifest + expected Markdown.
- `src/glossapi/`: Corpus pipeline, cleaners, and orchestration logic.
- `tests/test_pipeline_smoke.py`: Minimal regression entry point (uses the lightweight corpus).
- `docs/`: MkDocs site with onboarding, pipeline recipes, and configuration guides.

## Contributing
- Run `pytest tests/test_pipeline_smoke.py` for a fast end-to-end check.
- Regenerate the lightweight corpus via `generate_pdfs.py` and commit the updated PDFs + manifest together.
- Prefer `uv` or `pip` editable installs so Rust extensions rebuild locally.

Open an issue or PR if you spot drift between expected outputs and the pipeline, or if you have doc updates for the new Divio skeleton.

## License

This project is licensed under the [EUPL 1.2](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12).
