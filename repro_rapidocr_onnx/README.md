Repro: Docling + RapidOCR (ONNX, Greek)

Purpose

- Reproduce the working pipeline (Docling layout + RapidOCR ONNX) on any machine without relying on local paths or prior state.
- Capture the exact steps, scripts, and rationale that led to the final effective setup.

Quick Start

1) Create venv: `bash scripts/create_venv.sh`
2) Verify GPU providers: `python scripts/check_ort.py` → look for `CUDAExecutionProvider`
3) Patch Docling param mapping: `bash scripts/repatch_docling.sh`
4) Prepare keys file:
   - With your Paddle `inference.yml`: `bash scripts/prepare_keys.sh --yml /path/to/inference.yml --out /path/to/greek_keys.txt`
   - Or auto-download then extract: `bash scripts/prepare_keys.sh --download --out /path/to/greek_keys.txt`
5) Run ONNX pipeline:
   - Basic: `bash scripts/run_onnx.sh --det DET.onnx --rec REC.onnx --keys greek_keys.txt --in INPUT_PDFS --out OUTPUT_DIR [--device cuda:0]`
   - Use embedded text, OCR only bitmaps: add `--no-force-ocr`
   - Normalize output (default on): `--normalize-output|--no-normalize-output`
   - Optional math/code enrichment (Docling CodeFormula, GPU recommended): `--docling-formula [--formula-batch 8] [--docling-code]`

What’s in this folder (and how they relate)

- `greek_pdf_ocr.py`: the runner script. Uses Docling’s layout and RapidOCR with options for ONNX/Paddle. CLI flags:
  - `--backend onnxruntime|paddle`, `--onnx-det/--onnx-rec/--onnx-cls`, `--rec-keys`, `--images-scale`, `--text-score`, `--device`.
- `requirements.txt`: precise packages to install in `.venv_docling`.
- `ENVIRONMENT.md`: venv creation, provider checks, caches; what to avoid (CPU ORT, wrong RapidOCR flavor, GPU layout surprises).
- `MODELS.md`: what ONNX models you need, how to obtain/convert, and why we auto-locate the CLS model.
- `KEYS.md`: why and how to generate the Greek PP‑OCRv5 keys from Paddle `inference.yml`.
- `PATCHES.md`: the one-line Docling patch that maps `Rec.rec_keys_path` so keys are honored by RapidOCR.
- `RUN.md`: the step-by-step flow that ties the above together (create venv → patch → keys → run).
- `TROUBLESHOOTING.md`: symptoms → fixes for the issues we actually hit (factory masking, ORT CPU/GPU collision, CLS shape errors, missing keys).
- `scripts/`: automation for all of the above (venv creation, patching, keys extraction, ORT check, and final run).

History and rationale (why we did these steps)

- The Docling factory error (“No class found 'rapidocr'”) masked underlying RapidOCR init errors. We verified by constructing the model directly to see real exceptions.
- RapidOCR requires the Greek recognition keys; without them it tries to infer a dict URL and fails. We extract keys from the Paddle `inference.yml` to guarantee label alignment.
- Docling 2.48.0 passes `Rec.keys_path`; RapidOCR expects `Rec.rec_keys_path`. The provided patch ensures keys are wired correctly.
- Having `onnxruntime` CPU installed alongside `onnxruntime-gpu` confused provider reporting. We uninstall CPU ORT and stick to ORT GPU.
- A mismatched CLS ONNX caused input shape errors. We auto-locate RapidOCR’s packaged `ch_ppocr_mobile_v2.0_cls_infer.onnx`, which is shape-compatible.

Follow RUN.md for the exact sequence; use TROUBLESHOOTING.md if any step reports an error.
