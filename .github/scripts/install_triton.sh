#!/bin/bash
set -euo pipefail

python3 -m pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true

install_triton_from_wheelhouse() {
    local wheel_dir="$1"

    if [[ -z "${wheel_dir}" || ! -d "${wheel_dir}" ]]; then
        return 1
    fi

    local wheels=()
    shopt -s nullglob
    wheels=("${wheel_dir}"/triton*.whl)
    shopt -u nullglob

    if [[ "${#wheels[@]}" -eq 0 ]]; then
        echo "No triton wheels found in ${wheel_dir}"
        return 1
    fi

    echo "Installing triton from local wheelhouse: ${wheel_dir}"
    if python3 -m pip install --no-index --find-links "${wheel_dir}" triton; then
        return 0
    fi

    echo "Local triton wheel install failed; falling back to public index."
    return 1
}

TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-7.0.0/simple/"
ROCM_VERSION=$(dpkg -l rocm-core 2>/dev/null | awk '/^ii/{print $3}')
if [[ -n "$ROCM_VERSION" ]]; then
    ROCM_MAJOR_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f1,2)
    TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-${ROCM_MAJOR_MINOR}.0/simple/"
fi

TRITON_WHEEL_DIR=${TRITON_WHEEL_DIR:-}
if ! install_triton_from_wheelhouse "${TRITON_WHEEL_DIR}"; then
    echo "Installing triton from $TRITON_INDEX_URL"
    python3 -m pip install --extra-index-url "$TRITON_INDEX_URL" triton
fi

python3 - <<'PY'
import triton
from packaging.version import Version

if Version(triton.__version__) < Version("3.6.0"):
    raise SystemExit(f"triton>=3.6.0 is required, found {triton.__version__}")

print(f"Installed triton {triton.__version__}")
PY
