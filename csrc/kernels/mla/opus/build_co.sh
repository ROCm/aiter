#!/bin/bash
# Builds hsa/gfx950/mla_opus/opus_<name>.co from <name>.cu (every .cu here if no name is given)
# with the amdgpu-named-reg-pin-std LLVM (https://github.com/yuyzhang512/llvm-project), built with
#   ninja clang lld clang-offload-bundler llvm-objcopy
#
#   LLVM_BIN=/path/to/llvm-project/build/bin ./build_co.sh [name ...]
#
# A .cu whose first line is "// build_co.sh flags: <flags>" appends <flags> to the shared flags
# below; a repeated -mllvm option takes its last value.
set -euo pipefail

OPUS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER="$(cd "$OPUS/../../../.." && pwd)"
OUT_DIR="${OUT_DIR:-$AITER/hsa/gfx950/mla_opus}"

[ -x "${LLVM_BIN:-}/clang++" ] || { echo "set LLVM_BIN to the bin/ of an amdgpu-named-reg-pin-std LLVM build" >&2; exit 1; }
[ $# -gt 0 ] || set -- $(cd "$OPUS" && ls *.cu | sed 's/\.cu$//')

for name in "$@"; do
    out="$OUT_DIR/opus_$name.co"
    own_flags="$(sed -n '1s|^// build_co.sh flags: ||p' "$OPUS/$name.cu")"
    HIP_CLANG_PATH="$LLVM_BIN" /opt/rocm/bin/hipcc -x hip "$OPUS/$name.cu" \
        -I"$AITER/csrc/include" -I"$AITER/csrc/kernels/mla" \
        -std=c++20 -O3 --offload-arch=gfx950 -D__HIPCC_RTC__ \
        -ffast-math -fgpu-flush-denormals-to-zero -fno-offload-uniform-block \
        -mllvm --amdgpu-kernarg-preload-count=32 -mllvm --lsr-drop-solution=1 \
        -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false \
        -mllvm -enable-post-misched=0 \
        $own_flags -Rpass-analysis=kernel-resource-usage ${EXTRA_FLAGS:-} \
        --genco -o "$out" 2>&1 | { grep -E "error|Function Name|VGPRs:|AGPRs:|ScratchSize|Occupancy|SGPRs:" || true; }
    echo "wrote $out"
done
