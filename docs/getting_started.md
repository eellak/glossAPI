# Onboarding Guide

This guide gets a new GlossAPI contributor from clone → first extraction with minimal detours. Use it alongside the [Quickstart recipes](quickstart.md) once you're ready to explore specialised flows.

## Checklist

- Python 3.10+
- Recent `pip` (or `uv`) and a C/C++ toolchain for Rust wheels
- Optional: NVIDIA GPU with CUDA drivers for Docling acceleration

On fresh Linux hosts, make these assumptions explicit instead of relying on shell startup files:

- prefer a stable final CPython, not a prerelease distro build
- keep `~/.local/bin` on `PATH` if `uv` was installed with `pip install --user uv`
- keep `~/.cargo/bin` on `PATH` if Rust was installed with `rustup`

## Install GlossAPI

### Recommended setup

Use `./dependency_setup/setup_glossapi.sh --mode docling --run-tests` for the layout environment.
Install the default Greek OCR package with `python install_glossapi.py ocr --device cpu`
(or `mps` / `cuda126` / `cuda132`); see [OCR setup](ocr_plugin_migration.md).

### Option 1 — pip (evaluate quickly)

```bash
export PYTHONNOUSERSITE=1  # keep ~/.local packages out of the way
pip install glossapi
```

### Option 2 — local development (recommended)

```bash
git clone https://github.com/eellak/glossAPI.git
cd glossAPI
python -m venv .venv && source .venv/bin/activate
pip install -U pip maturin
pip install -e .
```

This builds the Rust extensions needed for `Corpus.clean()` and noise metrics. Re-run `pip install -e .` after pulling changes that touch Rust crates.

### Option 3 — conda on SageMaker / Amazon Linux

```bash
git clone https://github.com/eellak/glossAPI.git
cd glossAPI
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh
conda activate glossapi
```

The helper script provisions Python 3.10, installs Rust + `maturin`, and performs an editable install.

## GPU prerequisites (optional but recommended)

`setup_glossapi.sh` installs the Docling environment. If you are curating dependencies manually, make sure you:

- Select the PyTorch build that matches your driver/toolkit.
- Verify the providers with:

  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## First run (lightweight corpus)

```bash
python - <<'PY'
from pathlib import Path
from glossapi import Corpus

input_dir = Path("samples/lightweight_pdf_corpus/pdfs")
output_dir = Path("artifacts/lightweight_pdf_run")
output_dir.mkdir(parents=True, exist_ok=True)

corpus = Corpus(input_dir, output_dir)
corpus.extract(input_format="pdf")
PY
```

- Inspect `artifacts/lightweight_pdf_run/markdown/` and compare with `samples/lightweight_pdf_corpus/expected_outputs.json`.
- Run `pytest tests/test_pipeline_smoke.py` for a reproducible regression check tied to the same corpus.

## Next steps

- Jump into [Quickstart recipes](quickstart.md) for GPU OCR, Docling, and enrichment commands.
- Explore [Pipeline overview](pipeline.md) to understand each processing stage and emitted artifact.
- When ready to contribute docs, expand the placeholders in `docs/divio/`.
