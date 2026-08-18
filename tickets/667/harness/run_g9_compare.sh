#!/usr/bin/env bash
# SILOTIGER-667 G9 — one-command reproducible FlyDSL-vs-CK cold warp-decode compare (D6).
#
# Runs the of-record sweep and writes the checked-in artifact into tickets/667/.
# No clock-locking step: clocks can't be pinned on this gfx950 (only discrete
# {500,2400} SCLK levels; see plan D1). Instead the driver relies on D5's variance
# capture — the --repeats sweep records per-cell spread + a noisy flag, and samples
# the effective loaded SCLK into the artifact provenance header.
#
# Usage:
#   bash run_g9_compare.sh [options]
#     --gpu N          GPU index (default: env GPU, else 6)
#     --build-ck       (re)build the CK bench binary first via build_ck_bench.sh
#     --repeats N      D5 repeats for variance (default: 3)
#     --iters N        timed iters (default: 1000, of-record per D1)
#     --cold N         warmup/rotation iters (default: 20)
#     --out-prefix P   artifact path prefix (default: tickets/667/g9_compare)
#     --                everything after is passed through to compare.py
#
# Examples:
#   bash run_g9_compare.sh                      # full of-record sweep -> tickets/667/g9_compare.{md,csv}
#   bash run_g9_compare.sh --build-ck           # rebuild CK first, then sweep
#   bash run_g9_compare.sh --repeats 5 -- --shapes qwen3next --batches 1,8
#   bash run_g9_compare.sh --validate           # D7-lite numerical cross-check (no perf sweep)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"          # /workspaces/aiter
VENV_PY="${REPO}/flydsl_venv/bin/python"
COMPARE="${SCRIPT_DIR}/compare.py"
BUILD_CK="${SCRIPT_DIR}/build_ck_bench.sh"
CK_BIN="/workspaces/rocm-libraries-wdec/bench_ck_warp_decode"

GPU="${GPU:-6}"
BUILD_CK_FIRST=0
VALIDATE=0
REPEATS=3
ITERS=1000
COLD=20
OUT_PREFIX="${REPO}/tickets/667/g9_compare"
PASSTHRU=()

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)        GPU="$2"; shift 2 ;;
        --build-ck)   BUILD_CK_FIRST=1; shift ;;
        --validate)   VALIDATE=1; shift ;;
        --repeats)    REPEATS="$2"; shift 2 ;;
        --iters)      ITERS="$2"; shift 2 ;;
        --cold)       COLD="$2"; shift 2 ;;
        --out-prefix) OUT_PREFIX="$2"; shift 2 ;;
        --)           shift; PASSTHRU=("$@"); break ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ ! -x "${VENV_PY}" ]; then
    echo "error: flydsl venv python not found at ${VENV_PY}" >&2
    exit 1
fi

if [ "${BUILD_CK_FIRST}" -eq 1 ]; then
    echo "==> Building CK bench ..."
    bash "${BUILD_CK}"
fi

# D7-lite: dump CK FP8/BF16 kernel I/O on real host-quantized inputs, then rebuild
# the torch reference on the identical bytes and compare (cos / allclose). No perf
# sweep; exits non-zero if any kernel diverges.
if [ "${VALIDATE}" -eq 1 ]; then
    if [ ! -x "${CK_BIN}" ]; then
        echo "==> CK bench binary missing; building ..."
        bash "${BUILD_CK}"
    fi
    DUMP_DIR="${SCRIPT_DIR}/ck_validate_dump"
    mkdir -p "${DUMP_DIR}"
    rm -f "${DUMP_DIR}"/*.bin
    echo "==> D7-lite numerical cross-check: gpu=${GPU}  dump -> ${DUMP_DIR}"
    HIP_VISIBLE_DEVICES="${GPU}" CK_WD_VALIDATE=1 CK_WD_VALIDATE_DIR="${DUMP_DIR}" "${CK_BIN}"
    "${VENV_PY}" "${SCRIPT_DIR}/validate.py" --dir "${DUMP_DIR}"
    exit $?
fi

MD_OUT="${OUT_PREFIX}.md"
CSV_OUT="${OUT_PREFIX}.csv"

echo "==> G9 compare: gpu=${GPU} repeats=${REPEATS} iters=${ITERS} cold=${COLD}"
echo "    artifact -> ${MD_OUT} , ${CSV_OUT}"

HIP_VISIBLE_DEVICES="${GPU}" "${VENV_PY}" "${COMPARE}" \
    --iters "${ITERS}" --cold "${COLD}" --repeats "${REPEATS}" \
    --md-out "${MD_OUT}" --csv-out "${CSV_OUT}" \
    "${PASSTHRU[@]}"

echo ""
echo "==> Done. Artifact written:"
echo "    ${MD_OUT}"
echo "    ${CSV_OUT}"
