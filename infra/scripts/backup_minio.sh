#!/usr/bin/env sh
set -eu

ROOT_DIR="${BACKUP_ROOT:-/srv/mars-rec-sys-backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
TARGET_DIR="${ROOT_DIR}/minio/${STAMP}"
mkdir -p "${TARGET_DIR}"

docker run --rm \
  --network mlflow-network \
  -e MC_HOST_local="http://${MINIO_ROOT_USER}:${MINIO_ROOT_PASSWORD}@minio:9000" \
  -v "${TARGET_DIR}:/backup" \
  quay.io/minio/mc:latest \
  mirror "local/${MLFLOW_BUCKET_NAME}" /backup
