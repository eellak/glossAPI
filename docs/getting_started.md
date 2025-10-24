# Onboarding Guide

This guide gets a new GlossAPI contributor from clone → first extraction with minimal detours. Use it alongside the [Quickstart recipes](quickstart.md) once you're ready to explore specialised flows.

## Checklist

- Python 3.8+ (3.10 recommended)
- Recent `pip` (or `uv`) and a C/C++ toolchain for Rust wheels
- Optional: NVIDIA GPU with CUDA 12.x drivers for Docling/RapidOCR acceleration

## Install GlossAPI

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

- Install ONNX Runtime GPU and make sure the CPU wheel is absent:

  ```bash
  pip install onnxruntime-gpu==1.18.1
  pip uninstall -y onnxruntime || true
  ```

- For Docling layout + math enrichment on GPU:

  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
  ```

- Verify the providers:

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
