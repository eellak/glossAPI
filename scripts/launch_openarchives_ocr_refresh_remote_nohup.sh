#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <instance-name> <remote-root> [zone] [run-id]" >&2
  exit 1
fi

INSTANCE_NAME="$1"
REMOTE_ROOT="$2"
ZONE="${3:-${ZONE:-europe-central2-a}}"
RUN_ID="${4:-oa_refresh_$(date -u +%Y%m%dT%H%M%SZ)}"
PROJECT="${PROJECT:-eellak-glossapi-20251008}"
GCLOUD="/home/foivos/google-cloud-sdk/bin/gcloud"
RENDER_WORKERS="${RENDER_WORKERS:-32}"
NUM_THREADS="${NUM_THREADS:-32}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-8}"

REMOTE_RUN_ROOT="$REMOTE_ROOT/runs/$RUN_ID"
REMOTE_LOG_DIR="$REMOTE_ROOT/logs"
REMOTE_LOG="$REMOTE_LOG_DIR/$RUN_ID.log"
REMOTE_PID="$REMOTE_LOG_DIR/$RUN_ID.pid"

$GCLOUD compute ssh "$INSTANCE_NAME" --project "$PROJECT" --zone "$ZONE" --command "bash -lc 'set -euo pipefail; mkdir -p \"$REMOTE_LOG_DIR\" \"$REMOTE_RUN_ROOT\"; cd \"$REMOTE_ROOT\"; nohup bash -lc \"source .venv/bin/activate && export PYTHONPATH=$REMOTE_ROOT/glossAPI-development/src && python $REMOTE_ROOT/glossAPI-development/src/glossapi/scripts/openarchives_ocr_refresh.py run-end-to-end --repo-id glossAPI/openarchives.gr --hf-root $REMOTE_ROOT/hf_openarchives --ocr-manifest $REMOTE_ROOT/ocr_manifest.parquet --merged-markdown-root $REMOTE_ROOT/refresh_inputs/merged_markdown --run-root $REMOTE_RUN_ROOT --render-workers $RENDER_WORKERS --num-threads $NUM_THREADS --upload-workers $UPLOAD_WORKERS --verbose\" > \"$REMOTE_LOG\" 2>&1 < /dev/null & echo \$! > \"$REMOTE_PID\"; cat \"$REMOTE_PID\"'"

echo "launched run_id=$RUN_ID"
echo "remote_run_root=$REMOTE_RUN_ROOT"
echo "remote_log=$REMOTE_LOG"
echo "remote_pid_file=$REMOTE_PID"
