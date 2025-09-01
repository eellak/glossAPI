Environment Setup (Docling + RapidOCR ONNX)

Goal: Docling layout + RapidOCR (ONNXRuntime GPU) with Greek PP‑OCRv5 rec.

Create venv (no system path assumptions)

```
python3 -m venv .venv_docling
source .venv_docling/bin/activate
python -m pip install -U pip
python -m pip install -r repro_rapidocr_onnx/requirements.txt
```

Critical avoids

- Don’t install `onnxruntime` CPU alongside `onnxruntime-gpu`. If present, uninstall it:
  - `pip uninstall -y onnxruntime`
- Use `rapidocr_onnxruntime` (the ONNX flavor). The meta `rapidocr` alone is not sufficient for Docling integration.
- Keep `numpy<2` in this venv (ORT ABI compatibility).
- Keep layout on CPU unless you add a matching Torch CUDA; if enabling GPU layout later, consider `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`.
- Avoid local package shadowing (e.g., do not add this repo root to `PYTHONPATH` when testing unrelated packages).

Providers and caches

- Verify ORT GPU providers:
  - `python repro_rapidocr_onnx/scripts/check_ort.py` → should include `CUDAExecutionProvider`.
- Optional: set caches away from `$HOME` (customize to your system):
  - `TMPDIR=/path/to/tmp` `XDG_CACHE_HOME=//path/to/cache` `HF_HOME=/path/to/hf`

Post-install patch (Docling → RapidOCR) — auto-detects site-packages

Docling 2.48.0 passes `Rec.keys_path` to RapidOCR; RapidOCR expects `Rec.rec_keys_path`.
Reapply after reinstall:

```
bash repro_rapidocr_onnx/scripts/repatch_docling.sh
```

Optional: enable GPU layout

If you want Docling’s layout to run on GPU as well (OCR already uses ORT GPU):

```
source .venv_docling/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Optional stability vars if you see NCCL warnings:

```
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

Optional: enable Docling formula/code enrichment (LaTeX + code blocks)

- Enrichment is off by default. To enable with best performance on GPU:

```
source .venv_docling/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch torchvision
```

- Then add flags when running:

```
bash repro_rapidocr_onnx/scripts/run_onnx.sh ... \
  --docling-formula --formula-batch 8 [--docling-code]
```
```
