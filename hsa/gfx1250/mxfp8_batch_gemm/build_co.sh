#!/usr/bin/env bash
# Build all mxfp8_batch_gemm .co files from .s assembly sources.
#
# Usage:
#   ./build_co.sh                  # default: use shaders from poc_kl
#   SHADER_DIR=/path/to/shaders ./build_co.sh
#
# Prerequisites:
#   - amdclang++ or /opt/rocm/llvm/bin/clang++ available in PATH
#   - .s files already generated (run poc_kl/mi400/mxfp8fp4gemm/run.sh convert)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}"
SHADER_DIR="${SHADER_DIR:-/local_vol1_nobackup/zw/poc_kl/mi400/mxfp8fp4gemm/shaders}"
ARCH="gfx1250"

# Find compiler
if command -v amdclang++ &>/dev/null; then
    CXX="amdclang++"
elif [ -x /opt/rocm/llvm/bin/clang++ ]; then
    CXX="/opt/rocm/llvm/bin/clang++"
else
    echo "ERROR: amdclang++ or /opt/rocm/llvm/bin/clang++ not found" >&2
    exit 1
fi

echo "=== Building mxfp8_batch_gemm .co files ==="
echo "  Compiler:   ${CXX}"
echo "  Shader dir: ${SHADER_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Arch:       ${ARCH}"
echo ""

# Mapping: .s basename → .co output name (lowercase, aiter-style naming)
# Non-cluster variants only (cluster variants need AiterAsmKernel extension)
declare -A CO_NAMES=(
    ["MXFP8_GEMM_1TG_4W_64mx1_128nx4"]="mxfp8_batch_gemm_64x512"
    ["MXFP8_GEMM_1TG_4W_64mx1_128nx4_APRESHUFFLE"]="mxfp8_batch_gemm_64x512_apreshuffle"
    ["MXFP8_GEMM_1TG_4W_128mx2_128nx2"]="mxfp8_batch_gemm_256x256"
    ["MXFP8FP4_GEMM_1TG_4W_128mx2_128nx2"]="mxfp8fp4_batch_gemm_256x256"
    # Cluster variants (uncomment when cluster launch is supported):
    # ["MXFP8_GEMM_1TG_4W_64mx1_128nx4_CLUSTER1x4"]="mxfp8_batch_gemm_64x512_cluster1x4"
    # ["MXFP8_GEMM_1TG_4W_64mx1_128nx4_APRESHUFFLE_CLUSTER1x4"]="mxfp8_batch_gemm_64x512_apreshuffle_cluster1x4"
    # ["MXFP8_GEMM_1TG_4W_128mx2_128nx2_CLUSTER4x4"]="mxfp8_batch_gemm_256x256_cluster4x4"
    # ["MXFP8FP4_GEMM_1TG_4W_128mx2_128nx2_CLUSTER4x4"]="mxfp8fp4_batch_gemm_256x256_cluster4x4"
)

built=0
failed=0

for base in "${!CO_NAMES[@]}"; do
    s_file="${SHADER_DIR}/${base}.s"
    co_file="${OUTPUT_DIR}/${CO_NAMES[$base]}.co"

    if [ ! -f "${s_file}" ]; then
        echo "  SKIP  ${base}.s (not found)"
        continue
    fi

    echo -n "  BUILD ${base}.s -> $(basename ${co_file}) ... "
    if ${CXX} -x assembler -target amdgcn--amdhsa --offload-arch=${ARCH} \
        "${s_file}" -o "${co_file}" 2>/dev/null; then
        echo "OK"
        ((built++))
    else
        echo "FAILED"
        ((failed++))
    fi
done

echo ""
echo "=== Done: ${built} built, ${failed} failed ==="
