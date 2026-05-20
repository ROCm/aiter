#!/bin/bash
set -euo pipefail

python3 -m pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true

TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-7.0.0/simple/"
ROCM_VERSION=$(dpkg -l rocm-core 2>/dev/null | awk '/^ii/{print $3}')
if [[ -n "$ROCM_VERSION" ]]; then
    ROCM_MAJOR_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f1,2)
    TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-${ROCM_MAJOR_MINOR}.0/simple/"
fi

echo "Installing triton from $TRITON_INDEX_URL"
pip install --extra-index-url "$TRITON_INDEX_URL" triton

python3 - <<'PY'
import triton
from packaging.version import Version

if Version(triton.__version__) < Version("3.6.0"):
    raise SystemExit(f"triton>=3.6.0 is required, found {triton.__version__}")

print(f"Installed triton {triton.__version__}")
PY
