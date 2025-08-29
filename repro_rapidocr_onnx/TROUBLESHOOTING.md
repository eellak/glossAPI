Troubleshooting

Factory error: “No class found with the name 'rapidocr'”

- This message often masks underlying RapidOCR init errors.
- Reveal root cause by constructing `RapidOcrModel` directly in a REPL:
  - Import `RapidOcrModel` and pass options; examine the thrown exception.

RapidOCR not registered

- Ensure `import docling.models.rapid_ocr_model` happens before building `DocumentConverter`.
- Ensure `rapidocr_onnxruntime` is installed (not only `rapidocr`).
- Set `allow_external_plugins=True` in pipeline options.

ORT CPU vs GPU confusion

- If `onnxruntime` CPU is installed alongside `onnxruntime-gpu`, providers may be inconsistent.
- Uninstall CPU package: `pip uninstall -y onnxruntime`. Reinstall ORT GPU if needed.

Classifier shape error (80×160 expected)

- Symptom: ONNXRuntime INVALID_ARGUMENT on `x` input dims.
- Fix: use RapidOCR’s packaged `ch_ppocr_mobile_v2.0_cls_infer.onnx` for `--onnx-cls`.

Missing dict_url / keys file

- Symptom: `Missing key dict_url` from RapidOCR; factory error follows.
- Fix: generate keys from the Greek Paddle `inference.yml` and pass via `--rec-keys`.

Keys ignored

- Ensure the Docling patch is applied so RapidOCR receives `Rec.rec_keys_path`.

NCCL warnings (if enabling GPU layout)

- Set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` as needed.

