#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <instance-name> <remote-root> <run-id> [zone]" >&2
  exit 1
fi

INSTANCE_NAME="$1"
REMOTE_ROOT="$2"
RUN_ID="$3"
ZONE="${4:-${ZONE:-europe-central2-a}}"
PROJECT="${PROJECT:-eellak-glossapi-20251008}"
GCLOUD="/home/foivos/google-cloud-sdk/bin/gcloud"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-8}"

REMOTE_RUN_ROOT="$REMOTE_ROOT/runs/$RUN_ID"

$GCLOUD compute ssh "$INSTANCE_NAME" --project "$PROJECT" --zone "$ZONE" --command "bash -lc 'set -euo pipefail; cd \"$REMOTE_ROOT\"; source .venv/bin/activate; export PYTHONPATH=$REMOTE_ROOT/glossAPI-development/src; python $REMOTE_ROOT/glossAPI-development/src/glossapi/scripts/openarchives_ocr_refresh.py upload --staged-root $REMOTE_RUN_ROOT/staged_openarchives --repo-id glossAPI/openarchives.gr --num-workers $UPLOAD_WORKERS --verbose'"
