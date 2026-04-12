#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <instance-name> <remote-root> [zone]" >&2
  exit 1
fi

INSTANCE_NAME="$1"
REMOTE_ROOT="$2"
ZONE="${3:-${ZONE:-europe-central2-a}}"
PROJECT="${PROJECT:-eellak-glossapi-20251008}"

LOCAL_OCR_MANIFEST="/home/foivos/data/glossapi_work/analysis/openarchives_ocr_completion/release_refresh_20260403T180204Z/ocr_manifest.parquet"
GCLOUD="/home/foivos/google-cloud-sdk/bin/gcloud"
STAGE_DIR="$(mktemp -d /tmp/oa-ocr-refresh-stage.XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

REPO_TAR="$STAGE_DIR/glossapi-development.tar.gz"
MERGED_TAR="$STAGE_DIR/merged_markdown.tar.gz"

GZIP=-1 tar   --exclude='.git'   --exclude='.pytest_cache'   --exclude='__pycache__'   --exclude='*.pyc'   --exclude='.mypy_cache'   -czf "$REPO_TAR" -C /home/foivos glossAPI-development
GZIP=-1 tar -czf "$MERGED_TAR" -C /home/foivos/data/glossapi_work/analysis/openarchives_ocr_completion/release_refresh_20260403T180204Z merged_markdown

$GCLOUD compute ssh "$INSTANCE_NAME" --project "$PROJECT" --zone "$ZONE" --command "mkdir -p '$REMOTE_ROOT' && rm -f '$REMOTE_ROOT'/glossapi-development.tar.gz '$REMOTE_ROOT'/merged_markdown.tar.gz"
$GCLOUD compute scp --project "$PROJECT" --zone "$ZONE" "$REPO_TAR" "$INSTANCE_NAME:$REMOTE_ROOT/glossapi-development.tar.gz"
$GCLOUD compute scp --project "$PROJECT" --zone "$ZONE" "$LOCAL_OCR_MANIFEST" "$INSTANCE_NAME:$REMOTE_ROOT/ocr_manifest.parquet"
$GCLOUD compute scp --project "$PROJECT" --zone "$ZONE" "$MERGED_TAR" "$INSTANCE_NAME:$REMOTE_ROOT/merged_markdown.tar.gz"
$GCLOUD compute ssh "$INSTANCE_NAME" --project "$PROJECT" --zone "$ZONE" --command "cd '$REMOTE_ROOT' && tar -xzf glossapi-development.tar.gz && mkdir -p refresh_inputs && tar -xzf merged_markdown.tar.gz -C refresh_inputs && rm -f glossapi-development.tar.gz merged_markdown.tar.gz"

echo "staged repo and OCR inputs to $INSTANCE_NAME:$REMOTE_ROOT"
echo "remote_repo=$REMOTE_ROOT/glossAPI-development"
echo "remote_manifest=$REMOTE_ROOT/ocr_manifest.parquet"
echo "remote_merged_markdown=$REMOTE_ROOT/refresh_inputs/merged_markdown"
