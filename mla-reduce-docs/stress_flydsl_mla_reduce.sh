#!/usr/bin/env bash
# Stress regression for the FlyDSL MLA reduce multi-token decode fix.
#
# Reproduces the real end-to-end path that used to fault intermittently on
# decode_qlen > 1 with AITER_MLA_REDUCE_FLYDSL=1. The bug: the decode metadata
# hands the reduce a degenerate reduce_indptr = [0,0,0] (n_splits = 0) with an
# uninitialized reduce_final_map; the unguarded kernel read that garbage q-range
# and stored OOB -> intermittent GPU illegal memory access (~2/3 of runs).
# Fixed by the empty-tile guard in aiter/ops/flydsl/kernels/mla_reduce.py
# (skip tiles with n_splits <= 1). See
# mla-reduce-docs/MLA-reduce-flydsl-multitoken-decode-fix-report.md.
#
# Because the fault was intermittent, a single clean run proves nothing -- each
# config is run many times and faults are counted. Exits non-zero if ANY config
# faults even once, so this doubles as a pass/fail regression gate.
#
# Assumes the `mla_reduce_bench` container is up with the workspace bind-mounted
# at /aiter and the container-built JIT modules under /tmp/aiter_jit. Override
# the repeat counts with REPEATS=<n> to run a quicker/heavier sweep.

set -u

CONTAINER="${CONTAINER:-mla_reduce_bench}"

# test_cfg "<description>" <repeats> "<test_mla_sparse.py args>"
# Runs the config <repeats> times under the FlyDSL reduce path, counting GPU
# memory-access faults. Prints "<desc>: <clean> clean, <fault> fault (of N)".
overall_fault=0
test_cfg() {
    local desc="$1"; shift
    local reps="${REPEATS:-$1}"; shift
    local args="$1"; shift
    local fault=0 i out
    for i in $(seq 1 "$reps"); do
        out=$(docker exec \
            -e AITER_MLA_REDUCE_FLYDSL=1 \
            -e AITER_JIT_DIR=/tmp/aiter_jit \
            -w /aiter "$CONTAINER" \
            bash -lc "python3 op_tests/test_mla_sparse.py $args 2>&1")
        if echo "$out" | grep -qiE 'Memory access fault|GPU core'; then
            fault=$((fault + 1))
        fi
    done
    echo "$desc: $((reps - fault)) clean, $fault fault (of $reps)"
    overall_fault=$((overall_fault + fault))
}

test_cfg "bf16 qlen=2      " 20 "-n16,2 -b 1 -c 21  -k 512 -d bf16 -kvd bf16"
test_cfg "fp8  qlen=2      " 20 "-n16,2 -b 1 -c 21  -k 512 -d fp8  -kvd fp8"
test_cfg "bf16 qlen=3      "  8 "-n16,3 -b 1 -c 21  -k 512 -d bf16 -kvd bf16"
test_cfg "bf16 qlen=4      "  8 "-n16,4 -b 1 -c 21  -k 512 -d bf16 -kvd bf16"
test_cfg "bf16 qlen=2 b=4  "  8 "-n16,2 -b 4 -c 21  -k 512 -d bf16 -kvd bf16"
test_cfg "bf16 qlen=2 c=256"  8 "-n16,2 -b 1 -c 256 -k 512 -d bf16 -kvd bf16"

echo "---"
if [ "$overall_fault" -ne 0 ]; then
    echo "FAIL: $overall_fault total fault(s) -- the multi-token guard has regressed."
    exit 1
fi
echo "PASS: 0 faults across all configs."
