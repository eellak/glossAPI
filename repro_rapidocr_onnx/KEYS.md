Keys (recognizer labels)

RapidOCR’s ONNX recognizer needs the character dictionary that matches the rec model. For Greek PP‑OCRv5, generate the keys file from the Paddle inference config.

Steps

1) Ensure you have Greek Paddle inference folder unpacked:
   - `/mnt/data/models/paddlev5/el_PP-OCRv5_mobile_rec_infer/inference.yml`

2) Generate keys:

```
source .venv_docling/bin/activate
python repro_rapidocr_onnx/scripts/extract_keys.py \
  --in-yml /mnt/data/models/paddlev5/el_PP-OCRv5_mobile_rec_infer/inference.yml \
  --out /mnt/data/models/paddlev5/greek_ppocrv5_keys.txt
```

3) Use the keys file via `--rec-keys` in the runner.

Why this matters

- Without `rec_keys_path`, RapidOCR tries to resolve a dict URL for generic models and fails with a masked factory error.
- The extracted keys ensure Greek letters and diacritics align with your rec ONNX labels.

