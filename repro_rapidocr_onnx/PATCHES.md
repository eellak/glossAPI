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

Note on explicit injection path

- The repro runner now also sets `rec_keys_path` explicitly when using the explicit ONNX injection path. The patch remains recommended for users who rely on the factory path or run other Docling tools that construct the OCR engine via the factory.
