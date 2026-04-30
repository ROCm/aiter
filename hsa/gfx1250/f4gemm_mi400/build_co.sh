#!/usr/bin/env bash
# Build all f4gemm_mi400 .co files from .s assembly sources.
#
# Each F4GEMM sp3 emits a kernel symbol named "f4gemm". The CFG map in
# asm_f4gemm_mi400_configs.hpp uses (arch + knl_name) as a unique key, so we
# rename the kernel symbol per variant via sed before compilation. The
# rename target matches the knl_name column in f4gemm_mi400.csv.
#
# Expected input layout (one .s per (intype, subm, subn, apre)):
#   ${SHADER_DIR}/f4gemm_${intype_name}_${subm}x${subn}_apre${apre}.s
# where intype_name = "mxfp4" (intype=7) or "nvfp4" (intype=8).
#
# Generate these in poc_kl with `run.sh convert SUBM=.. SUBN=.. INTYPE=..
# A_PRESHUFFLE=.. SGPR_MODE=1` and rename / move the resulting .s into the
# expected filename.
#
# Usage:
#   ./build_co.sh                            # default SHADER_DIR
#   SHADER_DIR=/path/to/shaders ./build_co.sh
#   SHADER_DIR=... CSV=... ./build_co.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}"
CSV="${CSV:-${SCRIPT_DIR}/f4gemm_mi400.csv}"
SHADER_DIR="${SHADER_DIR:-${HOME}/Workspace/poc_kl/mi400/f4gemm/shaders}"
ARCH="gfx1250"

if command -v amdclang++ &>/dev/null; then
    CXX="amdclang++"
elif [ -x /opt/rocm/llvm/bin/clang++ ]; then
    CXX="/opt/rocm/llvm/bin/clang++"
else
    echo "ERROR: amdclang++ or /opt/rocm/llvm/bin/clang++ not found" >&2
    exit 1
fi

echo "=== Building f4gemm_mi400 .co files ==="
echo "  Compiler:   ${CXX}"
echo "  CSV:        ${CSV}"
echo "  Shader dir: ${SHADER_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Arch:       ${ARCH}"
echo ""

intype_name() {
    case "$1" in
        7) echo mxfp4 ;;
        8) echo nvfp4 ;;
        *) echo "unknown_intype_$1" ;;
    esac
}

built=0
skipped=0
failed=0

# Skip CSV header line
tail -n +2 "${CSV}" | while IFS=, read -r tile_m tile_n intype apre knl_name co_name; do
    [ -z "${tile_m}" ] && continue
    iname=$(intype_name "${intype}")
    s_basename="f4gemm_${iname}_${tile_m}x${tile_n}_apre${apre}"
    s_file="${SHADER_DIR}/${s_basename}.s"
    co_file="${OUTPUT_DIR}/${co_name}"

    if [ ! -f "${s_file}" ]; then
        printf "  SKIP  %-50s (missing %s)\n" "${co_name}" "${s_file}"
        skipped=$((skipped + 1))
        continue
    fi

    tmp_s="$(mktemp --suffix=.s)"
    # Rename the kernel symbol from "f4gemm" to the unique knl_name.
    # \b ensures we only match the standalone identifier, not substrings.
    sed -E "s/\bf4gemm\b/${knl_name}/g" "${s_file}" > "${tmp_s}"

    printf "  BUILD %-50s ... " "${co_name}"
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
echo "=== Done ==="
