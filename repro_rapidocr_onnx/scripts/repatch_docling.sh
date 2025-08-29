#!/usr/bin/env bash
set -euo pipefail

# Activate venv if present
if [[ -f .venv_docling/bin/activate ]]; then source .venv_docling/bin/activate; fi

SITE_FILE=$(python - <<'PY'
import inspect, docling.models.rapid_ocr_model as m
print(m.__file__)
PY
)

if [[ -z "$SITE_FILE" || ! -f "$SITE_FILE" ]]; then
  echo "Cannot locate docling.models.rapid_ocr_model.py (is the venv active?)" >&2
  exit 1
fi

if rg -n "Rec\.keys_path" "$SITE_FILE" >/dev/null; then
  sed -i "s/\"Rec.keys_path\"/\"Rec.rec_keys_path\"/" "$SITE_FILE"
  echo "Patched $SITE_FILE (Rec.rec_keys_path)"
else
  echo "Already patched or pattern not found in $SITE_FILE"
fi
