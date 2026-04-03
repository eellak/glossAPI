# Onboarding Guide

This guide gets a new GlossAPI contributor from clone → first extraction with minimal detours. Use it alongside the [Quickstart recipes](quickstart.md) once you're ready to explore specialised flows.

## Checklist

- Python 3.10+ (`3.12` recommended for the DeepSeek runtime)
- Recent `pip` (or `uv`) and a C/C++ toolchain for Rust wheels
- Optional: NVIDIA GPU with CUDA drivers for Docling/DeepSeek acceleration

On fresh Linux hosts, make these assumptions explicit instead of relying on shell startup files:

- prefer a stable final CPython, not a prerelease distro build
- keep `~/.local/bin` on `PATH` if `uv` was installed with `pip install --user uv`
- keep `~/.cargo/bin` on `PATH` if Rust was installed with `rustup`

## Install GlossAPI

### Recommended setup

Use `dependency_setup/setup_glossapi.sh` for the main Docling environment and `dependency_setup/setup_deepseek_uv.sh` for the OCR runtime. Examples:

```bash
# Main GlossAPI environment
./dependency_setup/setup_glossapi.sh --mode docling --venv dependency_setup/.venvs/docling --run-tests

# DeepSeek OCR on GPU (uv-managed, downloads DeepSeek-OCR-2 if requested)
./dependency_setup/setup_deepseek_uv.sh \
  --python /path/to/stable/python \
  --venv dependency_setup/.venvs/deepseek \
  --model-root /path/to/deepseek-ocr-2-model \
  --download-model \
  --run-tests --smoke-test
```

`setup_glossapi.sh --mode deepseek` delegates to the same uv-based installer. Inspect `dependency_setup/dependency_notes.md` for the current pins and validation runs. Both setup paths install GlossAPI and its Rust crates in editable mode so source changes are picked up immediately.
The dedicated DeepSeek uv environment is intentionally OCR-only: it installs `glossapi[deepseek]` and leaves Docling in the main environment.

On fresh GPU nodes, prefer a `uv`-managed stable Python such as:

```bash
~/.local/bin/uv python install 3.11.11
```

Then pass that interpreter explicitly to the setup scripts:

```bash
./dependency_setup/setup_glossapi.sh \
  --mode docling \
  --python /home/$USER/.local/share/uv/python/cpython-3.11.11-linux-x86_64-gnu/bin/python3.11 \
  --venv dependency_setup/.venvs/docling

./dependency_setup/setup_deepseek_uv.sh \
  --python /home/$USER/.local/share/uv/python/cpython-3.11.11-linux-x86_64-gnu/bin/python3.11 \
  --venv dependency_setup/.venvs/deepseek
```

**DeepSeek runtime checklist**
- Run `python -m glossapi.ocr.deepseek.preflight` from the DeepSeek venv to assert the real runtime is reachable.
- Run `python -m glossapi.scripts.deepseek_runtime_report` from the DeepSeek venv on fresh GPU nodes before ad hoc fixes. That captures the interpreter, CUDA wheel layout, and package versions used by the node.
- Force the real runtime and avoid stub fallback by setting:
  - `GLOSSAPI_DEEPSEEK_ALLOW_CLI=1`
  - `GLOSSAPI_DEEPSEEK_ALLOW_STUB=0`
  - `GLOSSAPI_DEEPSEEK_PYTHON=/path/to/deepseek/venv/bin/python`
  - `GLOSSAPI_DEEPSEEK_MODEL_DIR=/path/to/deepseek-ocr-2-model/DeepSeek-OCR-2`
- If `GLOSSAPI_DEEPSEEK_PYTHON` is unset, GlossAPI now searches for a repo-local version-pinned DeepSeek runtime under `dependency_setup/.venvs/deepseek*` before falling back to the generic `deepseek` alias and then the current process interpreter. Keep the env var set when you need an explicit override; broken explicit paths are treated as configuration errors, not silently ignored.
- Standard OCR defaults after setup:
  - `runtime_backend='vllm'`
  - `ocr_profile='markdown_grounded'`
  - `max_new_tokens=2048`
  - `repair_mode='auto'`
  - `scheduler='auto'`
  - `target_batch_pages=160`
- `flash-attn` is optional. The runner uses it when available and otherwise falls back to the Transformers `eager` attention implementation.
- Do not benchmark against an ad hoc DeepSeek venv and compare it to the validated `dependency_setup/.venvs/deepseek` results as if they were the same stack.

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

`setup_glossapi.sh` and `setup_deepseek_uv.sh` pull the required Torch wheels for the supported Docling and DeepSeek flows. If you are curating dependencies manually, make sure you:

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
