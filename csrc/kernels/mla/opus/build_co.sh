#!/bin/bash
# Builds hsa/gfx950/mla_opus/opus_<name>.co from <name>.cu (every .cu here if no name is given)
# with the amdgpu-named-reg-pin-std LLVM (https://github.com/yuyzhang512/llvm-project), built with
#   ninja clang lld clang-offload-bundler llvm-objcopy
#
#   PIN_LLVM=/path/to/llvm-project/build/bin ./build_co.sh [name ...]
set -euo pipefail

OPUS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER="$(cd "$OPUS/../../../.." && pwd)"
PIN_LLVM="${PIN_LLVM:-$HOME/llvm-project/build/bin}"
OUT_DIR="${OUT_DIR:-$AITER/hsa/gfx950/mla_opus}"

[ -x "$PIN_LLVM/clang++" ] || { echo "no pin toolchain at $PIN_LLVM" >&2; exit 1; }
[ $# -gt 0 ] || set -- $(cd "$OPUS" && ls *.cu | sed 's/\.cu$//')

for name in "$@"; do
    out="$OUT_DIR/opus_$name.co"
    HIP_CLANG_PATH="$PIN_LLVM" /opt/rocm/bin/hipcc -x hip "$OPUS/$name.cu" \
        -I"$AITER/csrc/include" -I"$AITER/csrc/kernels/mla" \
        -std=c++20 -O3 --offload-arch=gfx950 -D__HIPCC_RTC__ \
        -ffast-math -fgpu-flush-denormals-to-zero -fno-offload-uniform-block \
        -mllvm --amdgpu-kernarg-preload-count=32 -mllvm --lsr-drop-solution=1 \
        -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false \
        -mllvm -enable-post-misched=0 \
        -Rpass-analysis=kernel-resource-usage ${EXTRA_FLAGS:-} \
        --genco -o "$out" 2>&1 | { grep -E "error|Function Name|VGPRs:|AGPRs:|ScratchSize|Occupancy|SGPRs:" || true; }
    echo "wrote $out"
done
