#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --yml PATH_TO_inference.yml --out OUTPUT_KEYS

If --yml is missing, optionally download Greek PP-OCRv5 Paddle inference bundle and extract inference.yml.
EOF
}

YML=""; OUT=""; DL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yml) YML="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --download) DL=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$OUT" ]]; then usage; exit 1; fi

if [[ -z "$YML" && -n "$DL" ]]; then
  TMPD=$(mktemp -d)
  TAR="$TMPD/el_PP-OCRv5_mobile_rec_infer.tar"
  URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/el_PP-OCRv5_mobile_rec_infer.tar"
  echo "Downloading $URL ..."
  if command -v wget >/dev/null; then wget -q "$URL" -O "$TAR"; else curl -L "$URL" -o "$TAR"; fi
  mkdir -p "$TMPD/el_PP-OCRv5_mobile_rec_infer"
  tar -xf "$TAR" -C "$TMPD/el_PP-OCRv5_mobile_rec_infer"
  YML="$TMPD/el_PP-OCRv5_mobile_rec_infer/inference.yml"
fi

[[ -f "$YML" ]] || { echo "Missing --yml $YML" >&2; exit 1; }

source .venv_docling/bin/activate 2>/dev/null || true
python repro_rapidocr_onnx/scripts/extract_keys.py --in-yml "$YML" --out "$OUT"

