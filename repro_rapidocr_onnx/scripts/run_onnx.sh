#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --det DET_ONNX --rec REC_ONNX --keys KEYS_TXT --in INPUT_DIR --out OUTPUT_DIR [--device DEV] [--text-score X] [--images-scale S] [--no-force-ocr] [--normalize-output|--no-normalize-output] [--docling-formula] [--docling-code] [--formula-batch N]

Auto-locates the packaged RapidOCR CLS model and runs greek_pdf_ocr.py with ONNXRuntime backend.
EOF
}

DET=""; REC=""; KEYS=""; IN=""; OUT=""; DEV="cuda:0"; SCORE="0.45"; SCALE="1.25"; FORCE_OCR=1; NORM=1; FORMULA=0; CODE=0; FBATCH=""
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
    --no-force-ocr) FORCE_OCR=0; shift 1;;
    --normalize-output) NORM=1; shift 1;;
    --no-normalize-output) NORM=0; shift 1;;
    --docling-formula) FORMULA=1; shift 1;;
    --docling-code) CODE=1; shift 1;;
    --formula-batch) FBATCH="$2"; shift 2;;
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
  $( [[ $FORCE_OCR -eq 1 ]] && echo --force-ocr || echo --no-force-ocr ) \
  $( [[ $NORM -eq 1 ]] && echo --normalize-output || echo --no-normalize-output ) \
  $( [[ $FORMULA -eq 1 ]] && echo --docling-formula || true ) \
  $( [[ $CODE -eq 1 ]] && echo --docling-code || true ) \
  $( [[ -n "$FBATCH" ]] && echo --formula-batch "$FBATCH" || true ) \
  "$IN" "$OUT"
