#!/bin/bash

set -euo pipefail

TRITON_WHEEL_DIR="${1:-triton_wheels}"
mkdir -p "${TRITON_WHEEL_DIR}"

python3 -m pip config set global.retries 15
python3 -m pip config set global.timeout 120

TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-7.0.0/simple/"
ROCM_VERSION=$(dpkg -l rocm-core 2>/dev/null | awk '/^ii/{print $3}')
if [[ -n "${ROCM_VERSION}" ]]; then
    ROCM_MAJOR_MINOR=$(echo "${ROCM_VERSION}" | cut -d. -f1,2)
    TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-${ROCM_MAJOR_MINOR}.0/simple/"
fi

echo "Downloading triton wheel from ${TRITON_INDEX_URL} into ${TRITON_WHEEL_DIR}"
python3 -m pip download \
    --only-binary=:all: \
    --dest "${TRITON_WHEEL_DIR}" \
    --extra-index-url "${TRITON_INDEX_URL}" \
    triton

ls -lh "${TRITON_WHEEL_DIR}"
