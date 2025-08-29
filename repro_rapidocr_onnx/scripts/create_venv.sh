#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

python3 -m venv .venv_docling
source .venv_docling/bin/activate
python -m pip install -U pip
pip install -r repro_rapidocr_onnx/requirements.txt
# Avoid CPU ORT shadowing
pip uninstall -y onnxruntime || true
echo "Venv ready: .venv_docling"

