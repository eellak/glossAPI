#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --det DET_ONNX --rec REC_ONNX --keys KEYS_TXT --in INPUT_DIR --out OUTPUT_DIR [--device DEV] [--text-score X] [--images-scale S]

Auto-locates the packaged RapidOCR CLS model and runs greek_pdf_ocr.py with ONNXRuntime backend.
EOF
}

DET=""; REC=""; KEYS=""; IN=""; OUT=""; DEV="cuda:0"; SCORE="0.45"; SCALE="1.25"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --det) DET="$2"; shift 2;;
    --rec) REC="$2"; shift 2;;
    --keys) KEYS="$2"; shift 2;;
    --in) IN="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --device) DEV="$2"; shift 2;;
    --text-score) SCORE="$2"; shift 2;;
    --images-scale) SCALE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

[[ -f "$DET" && -f "$REC" && -f "$KEYS" && -d "$IN" ]] || { echo "Missing inputs" >&2; usage; exit 1; }
mkdir -p "$OUT"

# Activate venv if present
if [[ -f .venv_docling/bin/activate ]]; then source .venv_docling/bin/activate; fi

# Find packaged CLS ONNX
CLS=$(python - <<'PY'
import sys, pkgutil, pathlib
import rapidocr
base = pathlib.Path(rapidocr.__file__).parent / 'models'
cands = list(base.glob('*cls*_infer.onnx'))
for p in cands:
    if 'ch_ppocr_mobile_v2.0_cls_infer.onnx' in p.name:
        print(p)
        break
else:
    print(cands[0] if cands else '', end='')
PY
)
if [[ -z "$CLS" || ! -f "$CLS" ]]; then
  echo "Could not locate packaged CLS model in rapidocr/models" >&2
  exit 1
fi

PYTHONPATH=$(pwd) python greek_pdf_ocr.py \
  --backend onnxruntime \
  --device "$DEV" \
  --onnx-det "$DET" \
  --onnx-rec "$REC" \
  --onnx-cls "$CLS" \
  --rec-keys "$KEYS" \
  --text-score "$SCORE" \
  --images-scale "$SCALE" \
  "$IN" "$OUT"

