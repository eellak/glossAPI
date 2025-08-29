Decisions & Rationale

This documents the key choices we made to reach a stable, high-quality Docling + RapidOCR (ONNX) setup for Greek.

1) Use RapidOCR ONNXRuntime backend
- Why: Docling supports RapidOCR natively and we have PP-OCRv5 Greek rec ONNX. ONNXRuntime-GPU is widely available.
- Avoid: Installing generic `rapidocr` only; use `rapidocr_onnxruntime` to ensure the engine is present and loadable.

2) Keep layout on CPU in this repro
- Why: Avoids potential NCCL/Torch CUDA issues in varied environments. Docling layout still works, and OCR gains are large.
- If enabling GPU layout: install a matching Torch CUDA build and, if you see NCCL warnings, set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`.

3) Require explicit ONNX det/rec paths and auto-locate CLS
- Why: det/rec ONNX are user-provided or locally converted; CLS shape compatibility is tricky, so we auto-locate RapidOCR’s known-good CLS.
- Avoid: Using arbitrary CLS ONNX with mismatched input dims (causes INVALID_ARGUMENT errors in ORT).

4) Generate Greek keys from Paddle inference.yml
- Why: Recognition requires a character dictionary that matches the rec model labels. Extracting from `inference.yml` guarantees alignment.
- Avoid: Letting RapidOCR infer a dict URL; this fails for the Greek rec model and surfaces as a misleading factory error.

5) Patch Docling to pass `Rec.rec_keys_path`
- Why: Docling 2.48.0 uses `Rec.keys_path`; RapidOCR expects `Rec.rec_keys_path`. The one-line patch ensures keys are honored.
- How: `scripts/repatch_docling.sh` finds the installed file and patches in place; reapply after upgrades.

6) Avoid CPU ORT alongside ORT GPU
- Why: Having both `onnxruntime` and `onnxruntime-gpu` in the same venv can confuse provider detection and behavior.
- Fix: Uninstall `onnxruntime` CPU if present and reinstall `onnxruntime-gpu`.

7) Keep numpy<2
- Why: Best compatibility with ORT wheels and transitive deps we used.

8) Caches under user control
- Why: Environments vary; we don’t assume specific paths. The run scripts accept standard env vars if you want non-default cache locations.

