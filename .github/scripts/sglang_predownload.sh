#!/usr/bin/env bash
# Pre-stage a model into the persistent /models host cache before the SGLang
# downstream test runs. Mirrors the ATOM lane's decoupled "Download Models"
# step: the slow first download happens here (own job-step timeout), then the
# test step resolves from /models (via model_env_args) and never eats the
# download time. Safe to run concurrently on a shared node (flock) and
# idempotent across runs (.download-complete marker on the persistent mount).
#
# Usage: sglang_predownload.sh <MODEL_ID>
# Env:   HF_TOKEN (optional, for gated repos)
set -euo pipefail

MODEL_ID="${1:-}"
if [ -z "${MODEL_ID}" ]; then
  echo "usage: $0 <MODEL_ID>" >&2
  exit 2
fi

CACHE_ROOT="${MODEL_CACHE_ROOT:-/models}"
if [ ! -d "${CACHE_ROOT}" ]; then
  echo "[predownload] ${CACHE_ROOT} not present; skipping (test will fall back to HF)" >&2
  exit 0
fi

TARGET_DIR="${CACHE_ROOT}/${MODEL_ID}"
LOCK_ROOT="${CACHE_ROOT}/.locks"
mkdir -p "${LOCK_ROOT}"
SAFE_KEY="$(printf '%s' "${MODEL_ID}" | sed 's#[^A-Za-z0-9._-]#_#g')"
LOCK_FILE="${LOCK_ROOT}/${SAFE_KEY}.flock"

# Serialize concurrent downloaders of the same model on the shared node.
exec 9>"${LOCK_FILE}"
echo "[predownload] acquiring lock for ${MODEL_ID} ..."
flock 9

if [ -f "${TARGET_DIR}/.download-complete" ]; then
  echo "[predownload] cache hit: ${TARGET_DIR}"
  exit 0
fi

echo "[predownload] downloading ${MODEL_ID} -> ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"
MODEL_ID="${MODEL_ID}" TARGET_DIR="${TARGET_DIR}" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    os.environ["MODEL_ID"],
    local_dir=os.environ["TARGET_DIR"],
    max_workers=8,
)
PY

touch "${TARGET_DIR}/.download-complete"
echo "[predownload] done: ${TARGET_DIR}"
