#!/bin/bash
# Rebuilds ../f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co from this source (needs hipcc for gfx950).
set -e
cd "$(dirname "$0")"
hipcc -std=c++20 -O3 --offload-arch=gfx950 -mllvm -amdgpu-spill-vgpr-to-agpr=0 -mllvm -pragma-unroll-threshold=1000000 -mllvm -max-bytes-for-alignment=28 \
  --genco --no-gpu-bundle-output f4gemm_256x256.hip -o ../f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co
