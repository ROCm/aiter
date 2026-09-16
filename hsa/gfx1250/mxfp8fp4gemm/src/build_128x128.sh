#!/usr/bin/env bash
# Rebuild the registered K128/PF8 kernel without launching a GPU workload.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=f8gemm_bf16_mxfp8fp8_ABpreShuffle_128x128_4x4_ps
COMPILER=${AMDCLANG:-${ROCM_PATH:-/opt/rocm}/llvm/bin/amdclang++}
"$COMPILER" -x assembler -target amdgcn--amdhsa --offload-arch=gfx1250 \
    "$HERE/$BASE.s" -o "$HERE/../$BASE.co"
