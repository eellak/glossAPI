# Onboarding Guide

This guide gets a new GlossAPI contributor from clone → first extraction with minimal detours. Use it alongside the [Quickstart recipes](quickstart.md) once you're ready to explore specialised flows.

## Checklist

- Python 3.8+ (3.10 recommended)
- Recent `pip` (or `uv`) and a C/C++ toolchain for Rust wheels
- Optional: NVIDIA GPU with CUDA 12.x drivers for Docling/RapidOCR acceleration

## Install GlossAPI

### Recommended — mode-aware setup script

Use `dependency_setup/setup_glossapi.sh` to build an isolated virtualenv with the correct dependency set for vanilla, RapidOCR, or DeepSeek runs. Examples:

```bash
# Vanilla pipeline (CPU-only OCR)
./dependency_setup/setup_glossapi.sh --mode vanilla --venv dependency_setup/.venvs/vanilla --run-tests

# RapidOCR GPU stack
./dependency_setup/setup_glossapi.sh --mode rapidocr --venv dependency_setup/.venvs/rapidocr --run-tests

# DeepSeek OCR on GPU (expects weights under /path/to/deepseek-ocr/DeepSeek-OCR)
./dependency_setup/setup_glossapi.sh \
  --mode deepseek \
  --venv dependency_setup/.venvs/deepseek \
  --weights-dir /path/to/deepseek-ocr \
  --run-tests --smoke-test
```

Add `--download-deepseek` if you need the script to fetch weights via Hugging Face; otherwise it searches `${REPO_ROOT}/deepseek-ocr/DeepSeek-OCR` unless you override `--weights-dir`. Inspect `dependency_setup/dependency_notes.md` for the latest pins, caveats, and validation runs. The script installs GlossAPI and its Rust crates in editable mode so source changes are picked up immediately.

**DeepSeek runtime checklist**
- Run `python -m glossapi.ocr.deepseek.preflight` from the DeepSeek venv to assert the CLI can run (env vars, model dir, flashinfer, cc1plus, libjpeg).
- Force the real CLI and avoid stub fallback by setting:
  - `GLOSSAPI_DEEPSEEK_ALLOW_CLI=1`
  - `GLOSSAPI_DEEPSEEK_ALLOW_STUB=0`
  - `GLOSSAPI_DEEPSEEK_VLLM_SCRIPT=/path/to/deepseek-ocr/run_pdf_ocr_vllm.py`
  - `GLOSSAPI_DEEPSEEK_TEST_PYTHON=/path/to/deepseek/venv/bin/python`
  - `GLOSSAPI_DEEPSEEK_MODEL_DIR=/path/to/deepseek-ocr/DeepSeek-OCR`
  - `GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH=/path/to/libjpeg-turbo/lib`
- Install a CUDA toolkit with `nvcc` and set `CUDA_HOME` / prepend `$CUDA_HOME/bin` to `PATH` (FlashInfer/vLLM JIT expects it).
- If FlashInfer is unstable on your stack, disable it with `VLLM_USE_FLASHINFER=0` and `FLASHINFER_DISABLE=1`.
- Avoid FP8 KV cache issues by exporting `GLOSSAPI_DEEPSEEK_NO_FP8_KV=1`; tune VRAM use via `GLOSSAPI_DEEPSEEK_GPU_MEMORY_UTILIZATION=<0.5–0.9>`.
- Keep `LD_LIBRARY_PATH` pointing at the toolkit lib64 (e.g. `LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`).

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

The helper script provisions Python 3.10, installs Rust + `maturin`, performs an editable install, and applies the Docling RapidOCR patch automatically.

## GPU prerequisites (optional but recommended)

`setup_glossapi.sh` pulls the right CUDA/Torch/ONNX wheels for the RapidOCR and DeepSeek profiles. If you are curating dependencies manually, make sure you:

- Install the GPU build of ONNX Runtime (`onnxruntime-gpu`) and uninstall the CPU wheel.
- Select the PyTorch build that matches your driver/toolkit (the repository currently targets CUDA 12.8 for DeepSeek).
- Verify the providers with:

  ```bash
  python -c "import onnxruntime as ort; print(ort.get_available_providers())"
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## RapidOCR models & keys

GlossAPI ships the required ONNX models and Greek keys under `glossapi/models/rapidocr/{onnx,keys}`. To override them, set `GLOSSAPI_RAPIDOCR_ONNX_DIR` to a directory containing:

- `det/inference.onnx`
- `rec/inference.onnx`
- `cls/ch_ppocr_mobile_v2.0_cls_infer.onnx`
- `greek_ppocrv5_keys.txt`

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
