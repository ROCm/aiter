#!/bin/bash

set -ex

retry_cmd() {
    local max_attempts="$1"
    shift
    local attempt=1
    local rc=0

    while true; do
        if "$@"; then
            return 0
        fi
        rc=$?
        if [[ "$attempt" -ge "$max_attempts" ]]; then
            echo "Command failed after ${attempt} attempts: $*"
            return "$rc"
        fi
        local sleep_seconds=$((attempt * 20))
        echo "Attempt ${attempt}/${max_attempts} failed; retrying in ${sleep_seconds}s..."
        sleep "${sleep_seconds}"
        attempt=$((attempt + 1))
    done
}

echo
echo "==== ROCm Packages Installed ===="
dpkg -l | grep rocm || echo "No ROCm packages found."

echo
echo "==== Install dependencies and aiter ===="
git config --global --add safe.directory /workspace
pip config set global.retries 15
pip config set global.timeout 120
pip install --upgrade pandas pyzmq einops numpy==1.26.2 || {
    echo "WARNING: batch pip install failed, retrying packages individually..."
    pip install --upgrade pandas || true
    pip install --upgrade pyzmq || echo "WARNING: pyzmq unavailable (only needed by aiter.dist.shm_broadcast)"
    pip install --upgrade einops
    pip install --upgrade "numpy==1.26.2"
}
pip uninstall -y aiter || true
retry_cmd 3 pip install --upgrade \
    "pybind11>=3.0.1" \
    "setuptools>=45" \
    "setuptools_scm[toml]>=6.2" \
    wheel \
    packaging \
    psutil \
    "ninja>=1.11.1" \
    pandas \
    "flydsl==0.1.6"
pip install tabulate
retry_cmd 3 pip install --no-build-isolation -e .
./.github/scripts/install_triton.sh

# Read BUILD_TRITON env var, default to 1. If 1, install Triton; if 0, skip installation.
BUILD_TRITON=${BUILD_TRITON:-1}

if [[ "$BUILD_TRITON" == "1" ]]; then
    echo
    echo "==== Install triton ===="
    pip uninstall -y triton || true

    TRITON_WHEEL_DIR=${TRITON_WHEEL_DIR:-}
    if [[ -n "$TRITON_WHEEL_DIR" ]] && ls "$TRITON_WHEEL_DIR"/*.whl 1>/dev/null 2>&1; then
        echo "Installing triton from pre-built wheel in $TRITON_WHEEL_DIR"
        pip install "$TRITON_WHEEL_DIR"/*.whl
    else
        echo "Building triton from source..."
        # Pin triton to a known-good commit to avoid CI breakage from
        # upstream changes (e.g. AMD codegen regressions in triton-lang/triton).
        TRITON_COMMIT=${TRITON_COMMIT:-756afc06}
        git clone https://github.com/triton-lang/triton || true
        cd triton
        git checkout "$TRITON_COMMIT"
        pip install -r python/requirements.txt
        MAX_JOBS=64 pip --retries=10 --default-timeout=60 install .
        cd ..
    fi
    pip install filecheck
    # NetworkX is a dependency of Triton test selection script
    # `.github/scripts/select_triton_tests.py`.
    pip install networkx
else
    echo
    echo "[SKIP] Triton installation skipped because BUILD_TRITON=$BUILD_TRITON"
fi

echo
echo "==== Show installed packages ===="
pip list
