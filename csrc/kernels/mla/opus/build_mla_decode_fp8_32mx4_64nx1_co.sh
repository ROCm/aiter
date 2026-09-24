#!/bin/bash
# Builds hsa/gfx950/mla_opus/opus_mla_decode_fp8_32mx4_64nx1.co with the amdgpu-pin-op-dst
# toolchain (git@github.com:yuyzhang512/llvm-project.git, branch amdgpu-pin-op-dst), driven
# through hipcc so the ROCm device libs and --genco still apply.
#
#   PIN_LLVM=/path/to/llvm-project/build/bin ./build_mla_decode_fp8_32mx4_64nx1_co.sh
#
# The flags are the code generation aiter's JIT gives its in-tree kernels (the same set the
# 16mx4 code object is built with); -Rpass-analysis prints each entry's register budget.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER="$(cd "$HERE/../../../.." && pwd)"
PIN_LLVM="${PIN_LLVM:-$HOME/llvm-project/build/bin}"
OUT="${OUT:-$AITER/hsa/gfx950/mla_opus/opus_mla_decode_fp8_32mx4_64nx1.co}"

[ -x "$PIN_LLVM/clang++" ] || { echo "no pin toolchain at $PIN_LLVM" >&2; exit 1; }

HIP_CLANG_PATH="$PIN_LLVM" /opt/rocm/bin/hipcc -x hip "$HERE/mla_decode_fp8_32mx4_64nx1_co.hip" \
    -I"$AITER/csrc/include" -I"$AITER/csrc/kernels/mla" \
    -std=c++20 -O3 --offload-arch=gfx950 -D__HIPCC_RTC__ \
    -ffast-math -fgpu-flush-denormals-to-zero -fno-offload-uniform-block \
    -mllvm --amdgpu-kernarg-preload-count=32 -mllvm --lsr-drop-solution=1 \
    -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false \
    -mllvm -enable-post-misched=0 \
    -Rpass-analysis=kernel-resource-usage ${EXTRA_FLAGS:-} \
    --genco -o "$OUT" 2>&1 | grep -E "error|Function Name|VGPRs:|AGPRs:|ScratchSize|Occupancy|SGPRs:" || true

[ -s "$OUT" ] || { echo "build failed" >&2; exit 1; }
echo "wrote $OUT"
