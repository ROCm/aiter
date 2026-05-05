#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Drive op_tests/bench_zero_init_splitk_demo.py through all three configs.
#
# The bpreshuffle CKTile manifest (the C++ kernel-instance lookup table)
# is generated at compile time from the CSV pointed to by the env var
# AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE.  Switching to a CSV
# that selects a different set of kernelIds therefore requires a rebuild
# of module_gemm_a8w8_blockscale_bpreshuffle_cktile.  config1 and
# config2/config3 use different CSVs, so this driver deletes that
# module's .so + JIT build dir before config1 and again before
# config2 (config3 reuses the config2 build because both modes share
# the same with-SplitK CSV).
#
# Usage:
#   bash op_tests/run_zero_init_demo.sh [<results_dir>]
#
# Results dir defaults to op_tests/zero_init_demo_results/$(date +...).
set -euo pipefail

AITER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${1:-${AITER_ROOT}/op_tests/zero_init_demo_results/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RESULTS_DIR}"

NOSPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_nosplitk_gfx950.csv"
SPLITK_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_splitk_yz_gfx950.csv"
SHAPES_CSV="${AITER_ROOT}/aiter/configs/zero_init_demo/qwen3_next_80b_a3b_per1x128_untuned.csv"

ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-30}"
PYTHON="${PYTHON:-python}"
TRACE_DIR_FLAG=()
if [[ "${TRACE:-0}" == "1" ]]; then
    mkdir -p "${RESULTS_DIR}/traces"
    TRACE_DIR_FLAG=(--trace-dir "${RESULTS_DIR}/traces")
fi

JIT_DIR="${AITER_ROOT}/aiter/jit"
SO_FILE="${JIT_DIR}/module_gemm_a8w8_blockscale_bpreshuffle_cktile.so"
BUILD_DIR="${JIT_DIR}/build/module_gemm_a8w8_blockscale_bpreshuffle_cktile"

nuke_module() {
    echo "# nuking bpreshuffle CKTile module so it rebuilds against the active CSV"
    rm -f "${SO_FILE}"
    rm -rf "${BUILD_DIR}"
}

run_config() {
    local mode="$1"
    local csv="$2"
    local out_csv="${RESULTS_DIR}/${mode}.csv"
    local log="${RESULTS_DIR}/${mode}.log"
    echo "=========================================================="
    echo "# config: mode=${mode} csv=${csv}"
    echo "# results -> ${out_csv}"
    echo "# log     -> ${log}"
    echo "=========================================================="
    PYTHONPATH="${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="${csv}" \
        "${PYTHON}" "${AITER_ROOT}/op_tests/bench_zero_init_splitk_demo.py" \
        --mode "${mode}" \
        --tuned-csv "${csv}" \
        --shapes-csv "${SHAPES_CSV}" \
        --iters "${ITERS}" --warmup "${WARMUP}" \
        --out "${out_csv}" \
        "${TRACE_DIR_FLAG[@]}" \
        2>&1 | tee "${log}"
}

# config1: nosplitK baseline. The .so must be (re)built against the
# nosplitK CSV so the kernel manifest reflects those kernelIds.
nuke_module
run_config none "${NOSPLITK_CSV}"

# config2: splitK with kernel-side Y.zero_(). Different CSV -> rebuild.
nuke_module
run_config splitk "${SPLITK_CSV}"

# config3: splitK with producer-fused zero-init. Same CSV / same .so as
# config2; the only thing that flips is the runtime y_is_zeroed flag
# (and which kernel performs the zero fill).
run_config splitk_fused "${SPLITK_CSV}"

echo
echo "=========================================================="
echo "# Summary (median us, lower is better):"
echo "=========================================================="
"${PYTHON}" - <<PY
import csv, os
results_dir = "${RESULTS_DIR}"
modes = ["none", "splitk", "splitk_fused"]
data = {}
for mode in modes:
    path = os.path.join(results_dir, f"{mode}.csv")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for row in csv.DictReader(f):
            shape = (int(row["M"]), int(row["N"]), int(row["K"]))
            data.setdefault(shape, {})[mode] = float(row["median_us"])
print(f"{'shape':>15} | {'none':>9} | {'splitk':>9} | {'splitk_fused':>13} | {'splitk vs none':>15} | {'fused vs splitk':>16} | {'fused vs none':>14}")
print("-" * 110)
for shape in sorted(data):
    row = data[shape]
    n  = row.get("none", float("nan"))
    s  = row.get("splitk", float("nan"))
    sf = row.get("splitk_fused", float("nan"))
    def pct(a, b):
        if b is None or b != b or a != a:
            return "n/a"
        return f"{(a/b - 1) * 100:+6.1f}%"
    print(f"{shape!s:>15} | {n:>8.2f}u | {s:>8.2f}u | {sf:>12.2f}u | "
          f"{pct(s, n):>15} | {pct(sf, s):>16} | {pct(sf, n):>14}")
PY

echo
echo "# All artifacts saved under: ${RESULTS_DIR}"
