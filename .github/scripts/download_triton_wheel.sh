#!/bin/bash

set -euo pipefail

TRITON_WHEEL_DIR="${1:-triton_wheels}"
mkdir -p "${TRITON_WHEEL_DIR}"

python3 -m pip config set global.retries 15
python3 -m pip config set global.timeout 120

TRITON_DEFAULT_ROCM_VERSION="${TRITON_DEFAULT_ROCM_VERSION:-7.2.0}"
TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-${TRITON_DEFAULT_ROCM_VERSION}/simple/"
ROCM_VERSION=$(dpkg -l rocm-core 2>/dev/null | awk '/^ii/{print $3}' || true)
if [[ -z "${ROCM_VERSION}" ]]; then
    # RPM-based systems (e.g. rocm-core-7.2.0.70200-43.el8.x86_64 -> 7.2.0.70200)
    ROCM_VERSION=$(rpm -q --queryformat '%{VERSION}' rocm-core 2>/dev/null || true)
fi
if [[ -n "${ROCM_VERSION}" ]]; then
    ROCM_MAJOR_MINOR=$(echo "${ROCM_VERSION}" | cut -d. -f1,2)
    TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-${ROCM_MAJOR_MINOR}.0/simple/"
else
    echo "rocm-core not found; using default ROCm version ${TRITON_DEFAULT_ROCM_VERSION}"
fi

echo "Downloading triton wheel from ${TRITON_INDEX_URL} into ${TRITON_WHEEL_DIR}"
python3 -m pip download \
    --only-binary=:all: \
    --dest "${TRITON_WHEEL_DIR}" \
    --index-url "${TRITON_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "triton==3.7.0"

echo "Downloading triton-kernels wheel from ${TRITON_INDEX_URL} into ${TRITON_WHEEL_DIR}"
python3 -m pip download \
    --only-binary=:all: \
    --dest "${TRITON_WHEEL_DIR}" \
    --index-url "${TRITON_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "triton-kernels==1.0.0"

ls -lh "${TRITON_WHEEL_DIR}"
