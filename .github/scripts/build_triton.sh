#!/bin/bash

set -ex

# Install dependencies
pip install --upgrade
pip install pandas zmq einops numpy==1.26.2
python3 setup.py develop

# Install aiter
pip uninstall -y triton || true

# Clone the triton repository and install it
git clone --depth 1 https://github.com/triton-lang/triton || true
cd triton
pip install -r python/requirements.txt
pip install filecheck
pip install .

# Display installed packages
pip list
