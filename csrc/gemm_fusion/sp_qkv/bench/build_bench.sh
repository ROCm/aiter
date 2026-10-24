#!/usr/bin/env bash
# Build the R13 microbenches M2 (CU push) and M3 (BM32/BM64/BM128 split-K GEMM). SOURCE-ONLY recipe, never executed by the author.
# usage: BUILD_DIR=/fresh/dir bash build_bench.sh [m2|m3|all]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${HERE}/../../../.." && pwd)}"               # aiter checkout root (csrc/include/opus lives there)
OPUS_INC="${OPUS_INC:-${REPO_ROOT}/csrc/include}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
BUILD_DIR="${BUILD_DIR:?set BUILD_DIR to a fresh, non-existent output directory}"
[ -e "${BUILD_DIR}" ] && { echo "refuse: BUILD_DIR exists (${BUILD_DIR})" >&2; exit 1; }
bash -n "${BASH_SOURCE[0]}"
mkdir -p "${BUILD_DIR}"
FLAGS=( -I"${OPUS_INC}" -DOPUS_TILE_CONTAINER=0 -std=c++20 -O3 -fno-fast-math --offload-arch=gfx950 )
{ "${HIPCC}" --version; echo "FLAGS: ${FLAGS[*]}"; } | tee "${BUILD_DIR}/build-command.txt"
what="${1:-all}"
if [ "${what}" = "m2" ] || [ "${what}" = "all" ]; then
  "${HIPCC}" "${FLAGS[@]}" "${HERE}/m2_cu_push.cu" -o "${BUILD_DIR}/m2_cu_push"
  "${HIPCC}" "${FLAGS[@]}" --cuda-device-only -S "${HERE}/m2_cu_push.cu" -o "${BUILD_DIR}/m2_cu_push.gfx950.s"
fi
if [ "${what}" = "m3" ] || [ "${what}" = "all" ]; then
  "${HIPCC}" "${FLAGS[@]}" "${HERE}/m3_gemm_bm.cu" -o "${BUILD_DIR}/m3_gemm_bm"
  "${HIPCC}" "${FLAGS[@]}" --cuda-device-only -S "${HERE}/m3_gemm_bm.cu" -o "${BUILD_DIR}/m3_gemm_bm.gfx950.s"
  # ISA gates (see README): no scratch spill in any variant
  if grep -n 'private_segment_fixed_size: *[1-9]' "${BUILD_DIR}/m3_gemm_bm.gfx950.s" >/dev/null; then echo "GATE FAIL: scratch in M3 kernels" >&2; exit 4; fi
fi
sha256sum "${BUILD_DIR}"/m2_cu_push "${BUILD_DIR}"/m3_gemm_bm 2>/dev/null | tee "${BUILD_DIR}/binaries.sha256" || true
echo BENCH_BUILD_OK