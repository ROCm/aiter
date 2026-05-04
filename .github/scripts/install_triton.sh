#!/bin/bash
set -e

pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true

TRITON_INDEX_URL="https://pypi.amd.com/triton/rocm-7.0.0/simple/"
ROCM_VERSION=$(dpkg -l rocm-core 2>/dev/null | awk '/^ii/{print $3}')
if [[ -n "$ROCM_VERSION" ]]; then
    ROCM_MAJOR_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f1,2)
    TRITON_INDEX_URL="https://pypi.amd.com/triton/rocm-${ROCM_MAJOR_MINOR}.0/simple/"
fi

echo "Installing amd-triton from $TRITON_INDEX_URL"
pip install --extra-index-url "$TRITON_INDEX_URL" amd-triton
