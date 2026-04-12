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

REMOTE_LOG="$REMOTE_ROOT/logs/$RUN_ID.log"
REMOTE_PID_FILE="$REMOTE_ROOT/logs/$RUN_ID.pid"

$GCLOUD compute ssh "$INSTANCE_NAME" --project "$PROJECT" --zone "$ZONE" --command "bash -lc 'set -euo pipefail; PID=\$(cat \"$REMOTE_PID_FILE\"); echo pid=\$PID; ps -p \$PID -o pid=,etime=,pcpu=,pmem=,cmd= || true; echo ---LOG---; tail -n 50 \"$REMOTE_LOG\" || true'"
