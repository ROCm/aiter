#!/usr/bin/env bash
# Build all gfx1250 mxfp8fp4gemm .co files from the POC .s assembly sources.
#
# Each POC .s declares a uniquely-named kernel symbol "<variant>_KERNEL_FUNC"
# (e.g. MXFP8xFP8_GEMM_1TG_4W_128mx2_128nx2_APRESHUFFLE_CLUSTER4x4_PS_KERNEL_FUNC).
# The CFG map in asm_mxfp8fp4gemm_configs.hpp uses (arch + knl_name) as a unique
# key, and knl_name is now the mangled aiter symbol _ZN5aiter<len><base>E (see
# mxfp8fp4gemm.csv). So, like f4gemm, we rename the kernel symbol per variant via
# sed before compilation: <variant>_KERNEL_FUNC -> knl_name.
#
# The source .s filename / old symbol is derived from the row's (b_intype,
# tile_m, tile_n) -- there are only two registered tile shapes:
#   256x256 -> 128mx2_128nx2 + CLUSTER4x4   (cluster_x=4, cluster_y=4)
#   64x512  -> 64mx1_128nx4  + CLUSTER1x4   (cluster_x=4, cluster_y=1)
# and b_intype 0 -> FP8 (a8w8), 1 -> FP4 (a8w4); all rows are APRESHUFFLE + PS.
#
# Usage:
#   SHADER_DIR=/path/to/poc/mxfp8fp4gemm/shaders ./build_co.sh
#   SHADER_DIR=... CSV=... ./build_co.sh
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

# Map a registered (tile_m, tile_n) shape to the POC .s "<mx>_<nx>" + cluster tag.
tile_tag() {
    case "$1x$2" in
        256x256) echo "128mx2_128nx2 CLUSTER4x4" ;;
        64x512)  echo "64mx1_128nx4 CLUSTER1x4" ;;
        *)       echo "" ;;
    esac
}

built=0
skipped=0
failed=0

# Skip CSV header line; field order must match mxfp8fp4gemm.csv.
tail -n +2 "${CSV}" | while IFS=, read -r tile_m tile_n b_intype a_preshuffle cluster_x cluster_y persistent wg_max knl_name co_name; do
    [ -z "${tile_m}" ] && continue
    knl_name="$(echo "${knl_name}" | tr -d '[:space:]')"
    co_name="$(echo "${co_name}" | tr -d '[:space:]')"

    bw=$([ "${b_intype}" = "1" ] && echo 4 || echo 8)   # 0->FP8 (a8w8), 1->FP4 (a8w4)
    read -r mxnx cluster <<<"$(tile_tag "${tile_m}" "${tile_n}")"
    if [ -z "${mxnx}" ]; then
        printf "  SKIP  %-55s (no source mapping for ${tile_m}x${tile_n})\n" "${co_name}"
        skipped=$((skipped + 1))
        continue
    fi

    s_basename="MXFP8xFP${bw}_GEMM_1TG_4W_${mxnx}_APRESHUFFLE_${cluster}_PS"
    old_symbol="${s_basename}_KERNEL_FUNC"
    s_file="${SHADER_DIR}/${s_basename}.s"
    co_file="${OUTPUT_DIR}/${co_name}"

    if [ ! -f "${s_file}" ]; then
        printf "  SKIP  %-55s (missing %s)\n" "${co_name}" "${s_file}"
        skipped=$((skipped + 1))
        continue
    fi

    tmp_s="$(mktemp --suffix=.s)"
    # Rename the unique POC kernel symbol to the mangled aiter knl_name. \b keeps
    # the trailing ".kd" in ".symbol: <old>.kd" intact (it re-anchors at the dot).
    sed -E "s/\b${old_symbol}\b/${knl_name}/g" "${s_file}" > "${tmp_s}"

    printf "  BUILD %-55s ... " "${co_name}"
    if ${CXX} -x assembler -target amdgcn--amdhsa --offload-arch=${ARCH} \
        "${tmp_s}" -o "${co_file}" 2>/dev/null; then
        echo "OK"
        built=$((built + 1))
    else
        echo "FAILED"
        failed=$((failed + 1))
    fi
    rm -f "${tmp_s}"
done

echo ""
echo "=== Done (built=${built}, skipped=${skipped}, failed=${failed}) ==="
