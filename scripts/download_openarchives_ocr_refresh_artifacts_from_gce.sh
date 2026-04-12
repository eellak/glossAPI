#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <instance-name> <remote-root> <run-id> <local-dest-root> [zone]" >&2
  exit 1
fi

INSTANCE_NAME="$1"
REMOTE_ROOT="$2"
RUN_ID="$3"
LOCAL_DEST_ROOT="$4"
ZONE="${5:-${ZONE:-europe-central2-a}}"
PROJECT="${PROJECT:-eellak-glossapi-20251008}"
GCLOUD="/home/foivos/google-cloud-sdk/bin/gcloud"

REMOTE_RUN_ROOT="$REMOTE_ROOT/runs/$RUN_ID"
LOCAL_RUN_ROOT="$LOCAL_DEST_ROOT/$RUN_ID"
mkdir -p "$LOCAL_RUN_ROOT"

$GCLOUD compute scp --recurse --project "$PROJECT" --zone "$ZONE" \
  "$INSTANCE_NAME:$REMOTE_RUN_ROOT/debug" \
  "$INSTANCE_NAME:$REMOTE_RUN_ROOT/clean_markdown" \
  "$INSTANCE_NAME:$REMOTE_RUN_ROOT/target_summary.json" \
  "$INSTANCE_NAME:$REMOTE_RUN_ROOT/final_report.json" \
  "$INSTANCE_NAME:$REMOTE_RUN_ROOT/validation_summary.json" \
  "$INSTANCE_NAME:$REMOTE_RUN_ROOT/patch_summary.json" \
  "$INSTANCE_NAME:$REMOTE_RUN_ROOT/reevaluation_summary.json" \
  "$LOCAL_RUN_ROOT/"

echo "downloaded artifacts to $LOCAL_RUN_ROOT"
