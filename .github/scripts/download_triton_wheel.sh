#!/bin/bash

set -euo pipefail

TRITON_WHEEL_DIR="${1:-triton_wheels}"
mkdir -p "${TRITON_WHEEL_DIR}"

python3 -m pip config set global.retries 15
python3 -m pip config set global.timeout 120

TRITON_DEFAULT_ROCM_VERSION="${TRITON_DEFAULT_ROCM_VERSION:-7.2.0}"
TRITON_INDEX_URL="https://pypi.amd.com/triton/release_0914/rocm-${TRITON_DEFAULT_ROCM_VERSION}/simple/"
ROCM_VERSION=$(dpkg -l rocm-core 2>/dev/null | awk '/^ii/{print $3}' || true)
if [[ -z "${ROCM_VERSION}" ]]; then
    # RPM-based systems (e.g. rocm-core-7.2.0.70200-43.el8.x86_64 -> 7.2.0.70200)
    ROCM_VERSION=$(rpm -q --queryformat '%{VERSION}' rocm-core 2>/dev/null || true)
fi
if [[ -z "${ROCM_VERSION}" ]]; then
    ROCM_VERSION=$(python3 -m pip list --format=freeze 2>/dev/null | sed -n 's/^rocm==\([0-9][0-9.]*\).*/\1/p' || true)
fi
if [[ -n "${ROCM_VERSION}" ]]; then
    ROCM_MAJOR_MINOR=$(echo "${ROCM_VERSION}" | cut -d. -f1,2)
    TRITON_INDEX_URL="https://pypi.amd.com/triton/release_0914/rocm-${ROCM_MAJOR_MINOR}.0/simple/"
    if [[ "$(printf '%s\n' "${ROCM_MAJOR_MINOR}" 10.2 | sort -V | tail -n1)" != 10.2 ]] \
        && ! curl -fsS -o /dev/null "${TRITON_INDEX_URL}triton/" 2>/dev/null; then
        echo "No Triton index for ROCm ${ROCM_MAJOR_MINOR}; using ROCm 10.2.0"
        TRITON_INDEX_URL="https://pypi.amd.com/triton/release_0914/rocm-10.2.0/simple/"
    fi
else
    echo "rocm-core not found; using default ROCm version ${TRITON_DEFAULT_ROCM_VERSION}"
fi

echo "Downloading triton wheel from ${TRITON_INDEX_URL} into ${TRITON_WHEEL_DIR}"
python3 -m pip download \
    --no-deps \
    --only-binary=:all: \
    --dest "${TRITON_WHEEL_DIR}" \
    --index-url "${TRITON_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "triton==3.8.0"

echo "Downloading triton-kernels wheel from ${TRITON_INDEX_URL} into ${TRITON_WHEEL_DIR}"
python3 -m pip download \
    --no-deps \
    --only-binary=:all: \
    --dest "${TRITON_WHEEL_DIR}" \
    --index-url "${TRITON_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "triton-kernels==1.0.0"

ls -lh "${TRITON_WHEEL_DIR}"
