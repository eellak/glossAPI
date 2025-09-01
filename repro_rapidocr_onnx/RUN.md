Run Guide

Assumptions

- You have `.venv_docling` for Docling + RapidOCR ORT
- You have detection/recognition ONNX files (paths are arguments; no hard-coded locations)
- Optional: set cache env vars as desired (not required)

1) Create venv and install

```
python3 -m venv .venv_docling
source .venv_docling/bin/activate
pip install -U pip
pip install -r repro_rapidocr_onnx/requirements.txt
pip uninstall -y onnxruntime || true
```

2) Verify providers

```
python repro_rapidocr_onnx/scripts/check_ort.py
```

3) Patch Docling once

```
bash repro_rapidocr_onnx/scripts/repatch_docling.sh
```

4) Generate Greek keys

```
python repro_rapidocr_onnx/scripts/extract_keys.py \
  --in-yml /mnt/data/models/paddlev5/el_PP-OCRv5_mobile_rec_infer/inference.yml \
  --out /mnt/data/models/paddlev5/greek_ppocrv5_keys.txt
```

5) Run the pipeline (ONNX) — auto-locates packaged CLS model

```
bash repro_rapidocr_onnx/scripts/run_onnx.sh \
  --det /path/to/det/inference.onnx \
  --rec /path/to/rec/inference.onnx \
  --keys /path/to/greek_ppocrv5_keys.txt \
  --in /path/to/input_pdfs \
  --out /path/to/output_dir \
  --device cuda:0 \
  --text-score 0.45 \
  --images-scale 1.25 \
  --no-force-ocr \
  --normalize-output
```

Outputs

- Per-PDF `.md` and `.json` in the output directory

Optional: Enable GPU layout and Docling formula/code enrichment

Docling’s layout runs on GPU if a CUDA-enabled PyTorch is present and you pass `--device cuda:0` (already included above).

1) Install Torch CUDA in this venv (choose a CUDA build matching your driver; cu121 is a safe default):

```
source .venv_docling/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1
```

2) Verify and optionally enable enrichment flags:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bash repro_rapidocr_onnx/scripts/run_onnx.sh ... --docling-formula --formula-batch 8 [--docling-code]
```

3) (Optional) Stability envs and cache path:

```
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export DOCLING_CACHE_DIR=/path/to/docling_cache
```

4) Re-run `scripts/run_onnx.sh` with `--device cuda:0` (unchanged). Layout timings will appear under the `layout` key in `<stem>.metrics.json`.
