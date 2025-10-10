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
python3 setup.py develop

echo
echo "==== Install triton ===="
pip uninstall -y triton || true
git clone --depth=1 https://github.com/triton-lang/triton || true
cd triton
pip install -r python/requirements.txt
pip install filecheck
pip install .

echo
echo "==== Show installed packages ===="
pip list
