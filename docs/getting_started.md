# Getting Started

## Installation

```bash
export PYTHONNOUSERSITE=1  # prevent ~/.local packages from interfering
pip install glossapi
```

### Using conda on AWS SageMaker / Amazon Linux

If you are on a SageMaker instance where `conda` is the primary environment
manager, clone the repo and run the helper script to create a dedicated conda
env:

```bash
git clone https://github.com/your-org/glossAPI.git
cd glossAPI
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh

# Activate when needed
conda activate glossapi
```

The script provisions Python 3.10, installs Rust/maturin, and installs GlossAPI
in editable mode via `pip install -e .`, then builds the required Rust
extensions. Continue with the GPU prerequisites below inside the activated
environment.

### Build Rust extensions (required)

Whether you use a virtualenv or conda, run these commands once per environment
after installing dependencies:

```bash
python -m pip install "maturin>=1.5,<2.0"
python -m maturin develop --release --manifest-path rust/glossapi_rs_cleaner/Cargo.toml
python -m maturin develop --release --manifest-path rust/glossapi_rs_noise/Cargo.toml
```

Without these extensions `Corpus.clean()` and noise metrics will not work.

> **Note:** Remove any previously installed `glossapi_rs_cleaner`/
> `glossapi_rs_noise` wheels from your user site (e.g. `pip uninstall -y
> glossapi_rs_cleaner glossapi_rs_noise`) so the interpreter always imports the
> versions built inside your environment.

### GPU prerequisites

- NVIDIA GPU and recent driver (CUDA 12.x recommended).
- ONNXRuntime GPU (for OCR):
  - `pip install onnxruntime-gpu==1.18.1`
  - Ensure CPU ORT is NOT installed: `pip uninstall -y onnxruntime || true`
- Torch CUDA (for layout + math enrichment, optional):
  - `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1`

Verify:
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # must include CUDAExecutionProvider
python -c "import torch; print(torch.cuda.is_available())"
```

### Models & keys

GlossAPI ships packaged RapidOCR models/keys under `glossapi/models/rapidocr/{onnx,keys}`. To override, set `GLOSSAPI_RAPIDOCR_ONNX_DIR` to a directory containing:

- det/inference.onnx
- rec/inference.onnx
- cls/ch_ppocr_mobile_v2.0_cls_infer.onnx
- greek_ppocrv5_keys.txt

## First Run

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')

# Extract PDFs to Markdown (no OCR by default)
c.extract(input_format='pdf')

# Clean and compute quality metrics
c.clean()

# Re‑extract only bad files with GPU OCR
c.ocr(force=True)

# Section and annotate
c.section()
c.annotate()
```

See quickstart.md for condensed recipes (GPU OCR, multi‑GPU, math enrichment).
