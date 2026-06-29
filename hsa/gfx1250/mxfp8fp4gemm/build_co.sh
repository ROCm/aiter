#!/usr/bin/env bash
# Build all gfx1250 mxfp8fp4gemm .co files from the POC .s assembly sources.
#
# Each POC .s already declares a uniquely-named kernel symbol
#   <variant>_KERNEL_FUNC
# (e.g. MXFP8xFP8_GEMM_1TG_4W_128mx2_128nx2_APRESHUFFLE_CLUSTER4x4_PS_KERNEL_FUNC),
# which equals the knl_name column in mxfp8fp4gemm.csv. Because the symbols are
# already unique per variant, no symbol renaming is needed (unlike f4gemm) -- we
# just compile each .s to its co_name.
#
# Source .s filename is derived from knl_name by dropping the trailing
# "_KERNEL_FUNC", i.e. ${SHADER_DIR}/${knl_name%_KERNEL_FUNC}.s
#
# Usage:
#   SHADER_DIR=/path/to/poc/mxfp8fp4gemm/shaders ./build_co.sh
#
# Default SHADER_DIR points at the POC checkout next to this repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}"
CSV="${CSV:-${SCRIPT_DIR}/mxfp8fp4gemm.csv}"
SHADER_DIR="${SHADER_DIR:-/home/carhuang/yayang/Workspace/f8gemm/poc_kl_merg/poc_kl/mi400/mxfp8fp4gemm/shaders}"
ARCH="gfx1250"

if command -v amdclang++ &>/dev/null; then
    CXX="amdclang++"
elif [ -x /opt/rocm/llvm/bin/clang++ ]; then
    CXX="/opt/rocm/llvm/bin/clang++"
else
    echo "ERROR: amdclang++ or /opt/rocm/llvm/bin/clang++ not found" >&2
    exit 1
fi

echo "=== Building gfx1250 mxfp8fp4gemm .co files ==="
echo "  Compiler:   ${CXX}"
echo "  CSV:        ${CSV}"
echo "  Shader dir: ${SHADER_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Arch:       ${ARCH}"
echo ""

built=0
skipped=0
failed=0

# Skip CSV header line; field order must match mxfp8fp4gemm.csv.
tail -n +2 "${CSV}" | while IFS=, read -r tile_m tile_n b_intype a_preshuffle cluster_x cluster_y persistent wg_max knl_name co_name; do
    [ -z "${tile_m}" ] && continue
    knl_name="$(echo "${knl_name}" | tr -d '[:space:]')"
    co_name="$(echo "${co_name}" | tr -d '[:space:]')"
    s_basename="${knl_name%_KERNEL_FUNC}"
    s_file="${SHADER_DIR}/${s_basename}.s"
    co_file="${OUTPUT_DIR}/${co_name}"

    if [ ! -f "${s_file}" ]; then
        printf "  SKIP  %-40s (missing %s)\n" "${co_name}" "${s_file}"
        skipped=$((skipped + 1))
        continue
    fi

    printf "  BUILD %-40s ... " "${co_name}"
    if ${CXX} -x assembler -target amdgcn--amdhsa --offload-arch=${ARCH} \
        "${s_file}" -o "${co_file}"; then
        echo "OK"
        built=$((built + 1))
    else
        echo "FAILED"
        failed=$((failed + 1))
    fi
done

echo ""
echo "=== Done ==="
