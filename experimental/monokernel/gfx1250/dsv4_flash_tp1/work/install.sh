#!/bin/bash
set -euo pipefail
# Run inside the requested image with this directory mounted at /work.
cd /app/aiter
patch --dry-run -p1 < /work/aiter-hybrid.patch
cd /app/ATOM
patch --dry-run -p1 < /work/atom-hybrid.patch
cd /app/aiter
patch -p1 < /work/aiter-hybrid.patch
cd /app/ATOM
patch -p1 < /work/atom-hybrid.patch
python -m py_compile atom/models/deepseek_v4.py atom/model_ops/dsv4_hybrid.py
