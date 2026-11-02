#!/bin/bash
set -euo pipefail
cd /work/live-vector/ATOM
export PYTHONPATH=/work/live-vector/ATOM
export ATOM_VECTOR_UNFUSED=0 ATOM_VECTOR_CHECK=0
export HIP_VISIBLE_DEVICES=0 ATOM_DSV4_HYBRID=0 ATOM_DSV4_HYBRID_ATTN=0 ATOM_DSV4_HYBRID_GEMM=0 ATOM_DSV4_FLASH_MOE=0 ATOM_DSV4_MONO=0 ATOM_MONO_ENABLE=0 TRACE=0 FAKE_EPLB=0
export TRITON_CACHE_DIR=/work/flash/triton-cache
ulimit -c 0
bash /work/flash/serve.sh > /work/live-vector/hash-baseline-server.log 2>&1
