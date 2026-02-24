#!/bin/bash

set -ex

echo
echo "==== ROCm Packages Installed ===="
dpkg -l | grep rocm || echo "No ROCm packages found."

echo
echo "==== Install dependencies and aiter ===="
pip install --upgrade pandas zmq einops numpy==1.26.2
pip uninstall -y aiter || true
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
    git clone https://github.com/triton-lang/triton && cd triton && git checkout c147f098
    pip install -r python/requirements.txt
    pip install filecheck
    # NetworkX is a dependency of Triton test selection script
    # `.github/scripts/select_triton_tests.py`.
    pip install networkx
    MAX_JOBS=64 pip --retries=10 --default-timeout=60 install .
    cd ..
else
    echo
    echo "[SKIP] Triton installation skipped because BUILD_TRITON=$BUILD_TRITON"
fi

echo
echo "==== Show installed packages ===="
pip list
