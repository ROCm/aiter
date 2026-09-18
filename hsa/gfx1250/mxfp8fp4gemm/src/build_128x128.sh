#!/usr/bin/env bash
# Rebuild both registered K128/PF8 A layouts without launching a GPU workload.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COMPILER=${AMDCLANG:-${ROCM_PATH:-/opt/rocm}/llvm/bin/amdclang++}
for PRE in ABpreShuffle BpreShuffle; do
    BASE=f8gemm_bf16_mxfp8fp8_${PRE}_128x128_4x4_ps
    "$COMPILER" -x assembler -target amdgcn--amdhsa --offload-arch=gfx1250 \
        "$HERE/$BASE.s" -o "$HERE/../$BASE.co"
done
