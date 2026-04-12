#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <repo-root> <venv-path> <hf-root> <ocr-manifest> <merged-markdown-root> [run-root]" >&2
  exit 1
fi

REPO_ROOT="$1"
VENV_PATH="$2"
HF_ROOT="$3"
OCR_MANIFEST="$4"
MERGED_MARKDOWN_ROOT="$5"
RUN_ROOT="${6:-$PWD/openarchives_ocr_clean_refresh_run}"

mkdir -p "$RUN_ROOT"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$REPO_ROOT/src"

python "$REPO_ROOT/src/glossapi/scripts/openarchives_ocr_refresh.py" run-end-to-end \
  --repo-id glossAPI/openarchives.gr \
  --hf-root "$HF_ROOT" \
  --ocr-manifest "$OCR_MANIFEST" \
  --merged-markdown-root "$MERGED_MARKDOWN_ROOT" \
  --run-root "$RUN_ROOT" \
  --render-workers "$(nproc)" \
  --num-threads "$(nproc)" \
  --upload-workers 8 \
  --verbose
