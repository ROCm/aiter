#!/usr/bin/env bash
# Coverage runner for the gfx1250 mxfp8fp4 GEMM op test (a8w8 / a8w4).
#
# Usage:
#   ./op_tests/run_mxfp8fp4gemm_mi400.sh [func|perf|profile|cov] [a8w8|a8w4|both]
#     func     correctness only            (default)
#     perf     correctness + TFLOPS/TB-s table
#     profile  perf + torch profiler trace under ./aiter_logs
#     cov      func sweep then perf sweep   (full coverage)
#   second arg selects the dtype variant (default: both)
#
# Env overrides:
#   CONTAINER=yayang_f8gemm   docker container with the gfx1250 test env
#   APRE=1                    A-preshuffle (only apre=1 kernels are registered)
#   ENABLE_CK=0               use ck_tile_shim.h (required when the CK submodule
#                             is not checked out in the container)
#   FUNC_SHAPES / PERF_SHAPES override the shape lists below
#
# The script runs the test directly when torch+aiter import here (i.e. inside the
# container); otherwise it re-dispatches into $CONTAINER via `docker exec`.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TEST="op_tests/test_mxfp8fp4gemm_mi400.py"

MODE="${1:-func}"
INTYPE="${2:-both}"

CONTAINER="${CONTAINER:-yayang_f8gemm}"
APRE="${APRE:-1}"
GPU_ARCHS="${AITER_GPU_ARCHS:-gfx1250}"
CK="${ENABLE_CK:-0}"

# Shapes must satisfy at least one registered tile (the .cu heuristic picks):
#   256x256 (cluster4x4): M%1024==0, N%1024==0
#   64x512  (cluster1x4): M%64==0,   N%2048==0
#   all:    K%128==0
FUNC_SHAPES="${FUNC_SHAPES:-1024,1024,1024 1024,2048,1024 2048,2048,2048 4096,8192,1024 512,2048,1024 64,16384,1024 512,16384,1024}"
PERF_SHAPES="${PERF_SHAPES:-2048,2048,2048 4096,4096,4096 8192,8192,8192 4096,8192,2048 8192,8192,1024 64,16384,8192 512,16384,4096}"

# Decide whether to run here or inside the container.
if python -c "import torch, aiter" >/dev/null 2>&1; then
    RUN() { bash -lc "$1"; }
else
    echo "[run] torch/aiter not importable here -> docker exec ${CONTAINER}"
    RUN() { docker exec "${CONTAINER}" bash -lc "$1"; }
fi

run_sweep() {
    local mode="$1"; shift
    local shapes="$1"; shift
    echo ""
    echo "==================================================================="
    echo " mxfp8fp4gemm_mi400  mode=${mode}  intype=${INTYPE}  apre=${APRE}"
    echo "==================================================================="
    RUN "cd ${REPO} && AITER_GPU_ARCHS=${GPU_ARCHS} ENABLE_CK=${CK} \
        python ${TEST} --mode ${mode} --intype ${INTYPE} --apre ${APRE} -s ${shapes}"
}

case "${MODE}" in
    func)    run_sweep func    "${FUNC_SHAPES}" ;;
    perf)    run_sweep perf    "${PERF_SHAPES}" ;;
    profile) run_sweep profile "${PERF_SHAPES}" ;;
    cov)     run_sweep func "${FUNC_SHAPES}"; run_sweep perf "${PERF_SHAPES}" ;;
    *) echo "unknown mode '${MODE}' (use func|perf|profile|cov)" >&2; exit 1 ;;
esac
