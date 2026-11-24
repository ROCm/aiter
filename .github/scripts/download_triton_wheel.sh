#!/bin/bash

set -euo pipefail

TRITON_WHEEL_DIR="${1:-triton_wheels}"
mkdir -p "${TRITON_WHEEL_DIR}"

python3 -m pip config set global.retries 15
python3 -m pip config set global.timeout 120

# Same index lookup as install_triton.sh; fails if no AMD build fits this Python/ROCm.
TRITON_INDEX_URL=$(bash "$(dirname "${BASH_SOURCE[0]}")/install_triton.sh" --resolve)

echo "Downloading triton wheel from ${TRITON_INDEX_URL} into ${TRITON_WHEEL_DIR}"
python3 -m pip download \
    --no-deps \
    --only-binary=:all: \
    --dest "${TRITON_WHEEL_DIR}" \
    --index-url "${TRITON_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "triton==3.9.0"

echo "Downloading triton-kernels wheel from ${TRITON_INDEX_URL} into ${TRITON_WHEEL_DIR}"
python3 -m pip download \
    --no-deps \
    --only-binary=:all: \
    --dest "${TRITON_WHEEL_DIR}" \
    --index-url "${TRITON_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "triton-kernels==1.0.0"

ls -lh "${TRITON_WHEEL_DIR}"
