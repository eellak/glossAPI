# Getting Started

## Installation

```bash
pip install glossapi
```

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
