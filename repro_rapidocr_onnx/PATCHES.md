Docling rapidocr keys mapping (one-line patch)

Docling 2.48.0 sets `"Rec.keys_path"` in RapidOCR’s init params; RapidOCR expects `"Rec.rec_keys_path"`.

Patch location

- File: `.venv_docling/lib/python3.10/site-packages/docling/models/rapid_ocr_model.py`
- Change:
  - From: `"Rec.keys_path": self.options.rec_keys_path,`
  - To:   `"Rec.rec_keys_path": self.options.rec_keys_path,`

Reapply after reinstall

```
bash repro_rapidocr_onnx/scripts/repatch_docling.sh
```

Why it matters

- Without this, passing `--rec-keys` is ignored and RapidOCR errors when trying to infer a dict URL for the Greek model.

