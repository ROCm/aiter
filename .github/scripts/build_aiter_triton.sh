#!/bin/bash

set -ex

echo
echo "==== ROCm Packages Installed ===="
dpkg -l | grep rocm || echo "No ROCm packages found."

echo
echo "==== Install dependencies and aiter ===="
git config --global --add safe.directory /workspace
pip install --upgrade pandas zmq einops numpy==1.26.2
pip uninstall -y aiter || true
pip install --upgrade "pybind11>=3.0.1"
pip install --upgrade "ninja>=1.11.1"
pip install tabulate
pip install -e .

# Read BUILD_TRITON env var, default to 1. If 1, install Triton; if 0, skip installation.
BUILD_TRITON=${BUILD_TRITON:-1}

if [[ "$BUILD_TRITON" == "1" ]]; then
    echo
    echo "==== Install triton ===="
    bash ./.github/scripts/install_triton.sh
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
