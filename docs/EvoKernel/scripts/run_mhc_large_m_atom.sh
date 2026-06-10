#!/bin/bash
set -euo pipefail
REPO="${REPO:-/workspace/aiter}"
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
  -e PYTHONPATH=/workspace/aiter \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${REPO}:/workspace/aiter" \
  -w /workspace/aiter \
  rocm/atom:gfx950_latest \
  bash -lc '
    pip install -q flydsl==0.2.0 pandas tabulate einops pyyaml "triton>=3.6.0" 2>/dev/null || true
    pip install -q -e . 2>&1 | tail -5 || true
    python3 docs/EvoKernel/scripts/run_mhc_large_m_benchmark.py \
      -m 1024 2048 8192 65536 -n 4096 7168 \
      -o docs/EvoKernel/results/mhc_large_m_atom_gfx950.txt
  '
