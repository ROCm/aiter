#!/bin/bash

set -ex

# Install dependencies
pip install pandas zmq einops
pip install numpy==1.26.2

# Install aiter
pip uninstall -y triton || true
rm -rf /var/lib/jenkins/triton

# Clone the triton repository and install it
cd /var/lib/jenkins
git clone https://github.com/triton-lang/triton || true
cd triton
pip install -r python/requirements.txt
pip install filecheck
pip install .

# Display installed packages
pip list
