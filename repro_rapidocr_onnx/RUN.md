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
  --images-scale 1.25
```

Outputs

- Per-PDF `.md` and `.json` in the output directory
