#!/bin/bash
set -euo pipefail

if python3 - <<'PYCHECK' 2>/dev/null
import torch
from packaging.version import Version
exit(0 if Version(torch.__version__.split("+")[0].split("dev")[0]) < Version("2.9.1") else 1)
PYCHECK
then
    TRITON_INFO=$(python3 - <<'PYINFO' 2>/dev/null
try:
    import importlib.metadata as m
    for n in ["triton", "triton-rocm", "pytorch-triton-rocm", "pytorch-triton", "amd-triton"]:
        try:
            print(f"(keeping {n}=={m.version(n)})")
            break
        except Exception:
            pass
except Exception:
    pass
PYINFO
)
    echo "[aiter] torch < 2.9.1 detected, triton reinstall skipped for compatibility${TRITON_INFO:+ ${TRITON_INFO}}."
    echo "To use aiter-compatible triton, please upgrade torch to 2.9.1 or later."
    exit 0
fi

python3 -m pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true
python3 -m pip uninstall -y triton-kernels || true

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
    if ! python3 -m pip install --no-index --find-links "${wheel_dir}" triton; then
        echo "Local triton wheel install failed; falling back to public index."
        return 1
    fi

    echo "Installing triton-kernels from local wheelhouse: ${wheel_dir}"
    if ! python3 -m pip install --no-index --find-links "${wheel_dir}" triton-kernels; then
        echo "Local triton-kernels wheel install failed; falling back to public index."
        return 1
    fi

    return 0
}

TRITON_DEFAULT_ROCM_VERSION="${TRITON_DEFAULT_ROCM_VERSION:-7.2.0}"
TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-${TRITON_DEFAULT_ROCM_VERSION}/simple/"
ROCM_VERSION=$(dpkg -l rocm-core 2>/dev/null | awk '/^ii/{print $3}' || true)
if [[ -n "$ROCM_VERSION" ]]; then
    ROCM_MAJOR_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f1,2)
    TRITON_INDEX_URL="https://pypi.amd.com/triton/release_/rocm-${ROCM_MAJOR_MINOR}.0/simple/"
else
    echo "rocm-core not found; using default ROCm version ${TRITON_DEFAULT_ROCM_VERSION}"
fi

TRITON_WHEEL_DIR=${TRITON_WHEEL_DIR:-}
if ! install_triton_from_wheelhouse "${TRITON_WHEEL_DIR}"; then
    echo "Installing triton from $TRITON_INDEX_URL"
    python3 -m pip install --extra-index-url "$TRITON_INDEX_URL" "triton==3.7.0"

    echo "Installing triton-kernels from $TRITON_INDEX_URL"
    python3 -m pip install --extra-index-url "$TRITON_INDEX_URL" "triton-kernels==1.0.0"
fi

python3 - <<'PY'
import triton
from packaging.version import Version

if Version(triton.__version__) < Version("3.6.0"):
    raise SystemExit(f"triton>=3.6.0 is required, found {triton.__version__}")

print(f"Installed triton {triton.__version__}")
PY
