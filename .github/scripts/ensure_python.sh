#!/bin/bash
set -euo pipefail

# ROCm PyTorch images may only provide python3; CI also invokes python.
if ! command -v python >/dev/null 2>&1; then
    ln -s /usr/bin/python3 /usr/local/bin/python
fi
