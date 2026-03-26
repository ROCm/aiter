#!/bin/bash

set -ex

echo
echo "==== ROCm Packages Installed ===="
dpkg -l | grep rocm || echo "No ROCm packages found."

echo
echo "==== Install dependencies and aiter ===="
pip install --upgrade pandas zmq einops numpy==1.26.2
pip uninstall -y aiter || true
pip install --upgrade "setuptools_scm<9"
pip install --upgrade "pybind11>=3.0.1"
pip install --upgrade "ninja>=1.11.1"
pip install tabulate
python3 setup.py develop

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
