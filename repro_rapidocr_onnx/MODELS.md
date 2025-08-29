Models (ONNX)

You need three ONNX models:

- Detection (det): PP‑OCRv5 det ONNX
- Classification (cls): text orientation classifier ONNX
- Recognition (rec): PP‑OCRv5 Greek rec ONNX

Paths used in our working setup

- Det: `/mnt/data/models/paddlev5/det_onnx/inference.onnx`
- Rec: `/mnt/data/models/paddlev5/rec_onnx/inference.onnx` (Greek v5)
- Cls: `.venv_docling/lib/python3.10/site-packages/rapidocr/models/ch_ppocr_mobile_v2.0_cls_infer.onnx`

Notes

- The classifier must match expected input dims; RapidOCR’s bundled `ch_ppocr_mobile_v2.0_cls_infer.onnx` is compatible.
- If your own cls ONNX fails with input shape errors, switch to the bundled one.

Conversion options for rec/det

- Option A: Use preconverted ONNX you already have in `/mnt/data/models/paddlev5/*_onnx/`.
- Option B: Convert Paddle inference → ONNX using RapidAI’s PaddleOCRModelConvert or Paddle’s official tooling. High-level steps:
  1) Obtain Paddle inference folders for PP‑OCRv5 models (det server/mobile, Greek rec). See `find_v5models.md` and `how_toget_models.md`.
  2) Use a converter script (e.g., PaddleOCRModelConvert) to export `inference.onnx` for each.
  3) Verify the ONNX loads with onnxruntime and matches expected input sizes.

Detector (server) vs mobile

- Server det is more accurate (slower); mobile det is faster. Use either; both work with RapidOCR.

Sanity checks

```
python - << 'PY'
import onnxruntime as ort
for p in [
    '/mnt/data/models/paddlev5/det_onnx/inference.onnx',
    '/mnt/data/models/paddlev5/rec_onnx/inference.onnx',
    '/mnt/data/greek_paddleocr_pipeline/.venv_docling/lib/python3.10/site-packages/rapidocr/models/ch_ppocr_mobile_v2.0_cls_infer.onnx',
]:
    s = ort.InferenceSession(p, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    print('OK:', p)
PY
```

