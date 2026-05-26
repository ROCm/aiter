#!/bin/bash

set -euo pipefail

TRITON_WHEEL_DIR="${1:-triton_wheels}"
mkdir -p "${TRITON_WHEEL_DIR}"

python3 -m pip config set global.retries 15
python3 -m pip config set global.timeout 120

TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-7.0.0/simple/"
ROCM_VERSION=$(dpkg -l rocm-core 2>/dev/null | awk '/^ii/{print $3}')
ROCM_MAJOR_MINOR="7.0"
if [[ -n "${ROCM_VERSION}" ]]; then
    ROCM_MAJOR_MINOR=$(echo "${ROCM_VERSION}" | cut -d. -f1,2)
    TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-${ROCM_MAJOR_MINOR}.0/simple/"
fi
PYTHON_TAG=$(python3 -c 'import sys; print(f"py{sys.version_info.major}_{sys.version_info.minor}")')

echo "Downloading triton wheel from ${TRITON_INDEX_URL} into ${TRITON_WHEEL_DIR}"
python3 -m pip download \
    --only-binary=:all: \
    --dest "${TRITON_WHEEL_DIR}" \
    --index-url "${TRITON_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    triton

{
    echo "ROCM_MAJOR_MINOR=${ROCM_MAJOR_MINOR}"
    echo "PYTHON_TAG=${PYTHON_TAG}"
} > "${TRITON_WHEEL_DIR}/triton-wheel-metadata.env"

ls -lh "${TRITON_WHEEL_DIR}"
