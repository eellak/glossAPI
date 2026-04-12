#!/usr/bin/env bash
set -euo pipefail

INSTANCE_NAME="${1:-oa-ocr-clean-refresh}"
ZONE="${ZONE:-europe-central2-a}"
MACHINE_TYPE="${MACHINE_TYPE:-n4-standard-32}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-500}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"
PROJECT="${PROJECT:-eellak-glossapi-20251008}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
SERVICE_ACCOUNT_SCOPES="${SERVICE_ACCOUNT_SCOPES:-https://www.googleapis.com/auth/cloud-platform}"

/home/foivos/google-cloud-sdk/bin/gcloud compute instances create "$INSTANCE_NAME" \
  --project "$PROJECT" \
  --zone "$ZONE" \
  --machine-type "$MACHINE_TYPE" \
  --boot-disk-size "${BOOT_DISK_SIZE_GB}GB" \
  --boot-disk-type "$BOOT_DISK_TYPE" \
  --image-family "$IMAGE_FAMILY" \
  --image-project "$IMAGE_PROJECT" \
  --scopes "$SERVICE_ACCOUNT_SCOPES" \
  --metadata enable-oslogin=FALSE

echo "created instance=$INSTANCE_NAME zone=$ZONE project=$PROJECT"
