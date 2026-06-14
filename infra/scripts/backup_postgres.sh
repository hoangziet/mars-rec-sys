#!/usr/bin/env sh
set -eu

ROOT_DIR="${BACKUP_ROOT:-/srv/mars-rec-sys-backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
TARGET_DIR="${ROOT_DIR}/postgres"
mkdir -p "${TARGET_DIR}"

docker exec mlflow-postgres pg_dump -U "${POSTGRES_USER}" "${POSTGRES_DB}" > "${TARGET_DIR}/mlflow-${STAMP}.sql"
