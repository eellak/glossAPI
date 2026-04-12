#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <instance-name> [zone]" >&2
  exit 1
fi

INSTANCE_NAME="$1"
ZONE="${2:-${ZONE:-europe-central2-a}}"
PROJECT="${PROJECT:-eellak-glossapi-20251008}"
PYTHON_BIN="${PYTHON_BIN:-/home/foivos/data/glossapi_work/.venv/bin/python}"
GCLOUD="/home/foivos/google-cloud-sdk/bin/gcloud"

TOKEN="$($PYTHON_BIN - <<'PY'
from huggingface_hub.utils import get_token
value = get_token()
if not value:
    raise SystemExit(1)
print(value)
PY
)"

TMP_DIR="$(mktemp -d /tmp/oa-hf-token.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
TOKEN_FILE="$TMP_DIR/token"
printf '%s' "$TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"

$GCLOUD compute ssh "$INSTANCE_NAME" --project "$PROJECT" --zone "$ZONE" --command 'mkdir -p ~/.cache/huggingface && chmod 700 ~/.cache/huggingface'
$GCLOUD compute scp --project "$PROJECT" --zone "$ZONE" "$TOKEN_FILE" "$INSTANCE_NAME:~/.cache/huggingface/token"
$GCLOUD compute ssh "$INSTANCE_NAME" --project "$PROJECT" --zone "$ZONE" --command 'chmod 600 ~/.cache/huggingface/token'

echo "configured Hugging Face token on $INSTANCE_NAME"
