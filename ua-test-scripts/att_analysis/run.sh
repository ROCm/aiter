#!/usr/bin/env bash
# One-command, repeatable FA4 ATT overlap analysis.
#
#   att_analysis/run.sh <dtype> <batch> <sq> <sk> [simd] [iters]
#
# Steps:
#   1. (optional) rebuild+deploy the UA module WITH DWARF line tables so the
#      trace carries ISA->C++ source (set LINETABLES=1; needed once after any
#      source change -- -gline-tables-only does not change -O3 codegen). When
#      SLIM=1 (default) the rebuild compiles ONLY the traced instance, which is
#      both fast to compile (4 TUs vs 125) and fast to trace (rocprofv3 ATT must
#      disassemble every compiled kernel -- the full module is 50MB/610 kernels
#      and costs ~110s of pure disassembly per collection; one instance is ~1s).
#   2. collect an Advanced Thread Trace for the UA kernel (rocprof_att_prefill.sh).
#   3. render the two-warpgroup overlap timeline + Markdown report.
#
# Env:
#   LINETABLES=1   rebuild before tracing (embed source). Needed once after any
#                  source change, and once to switch the deployed .so to/from SLIM.
#   SLIM=1         (default) build only the traced instance. SLIM=0 builds the
#                  full module (needed for correctness/perf sweeps over shapes).
#   MASK=0|2       0=non-causal (default), 2=causal -> selects mask/nmask instance.
#   GPU=2          which GPU
#   SIMD, ITERS    forwarded to the report (also positional 5th/6th args)
#   TAG            run dir name (default att_lt_d128_<dtype>_b<batch>_sq<sq>_sk<sk>)
#
# NOTE: after a SLIM build the deployed module_unified_attention.so contains only
# this one instance; other shapes/dtypes/masks fail with "no matching kernel".
# Rebuild with SLIM=0 LINETABLES=1 (or AITER_REBUILD=1) to restore the full module.
set -euo pipefail

DTYPE="${1:?dtype: fp8 or bf16}"; BATCH="${2:?batch}"; SQ="${3:?sq}"; SK="${4:?sk}"
SIMD="${5:-${SIMD:-0}}"; ITERS="${6:-${ITERS:-3}}"
GPU="${GPU:-2}"
D="${D:-128}"; MASK="${MASK:-0}"; SLIM="${SLIM:-1}"

# Fast single-SIMD collection (the overlap only needs the two co-resident waves
# on one SIMD). Tracing one SIMD shrinks the ATT blob ~4x and the decode with
# it. Override SIMD_MASK=0xF to capture all four SIMDs.
export SIMD_MASK="${SIMD_MASK:-0x1}"
export MASK CONTIG=1   # forwarded to rocprof_att_prefill.sh (contiguous, non/causal)
# ATT traces a single dispatch (--att-consecutive-kernels 1), so the 101-iter
# @perftest loop is wasted work here: the FIRST UA launch under rocprofv3 is the
# one captured. We only need the kernel compiled (import) + at least one launch.
# Floor is warmup=1 + iters=2 because @perftest's stats path asserts num_iters>1.
export AITER_PERF_ITERS="${AITER_PERF_ITERS:-2}"
export AITER_PERF_WARMUP="${AITER_PERF_WARMUP:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$(dirname "$HERE")"            # ua-test-scripts
AITER_ROOT="$(dirname "$SCRIPTS")"      # repo root
TAG="${TAG:-att_lt_d128_${DTYPE}_b${BATCH}_sq${SQ}_sk${SK}}"
RUN_DIR="$SCRIPTS/rocprof_analysis/runs/$TAG"

# The single contiguous (nopage) instance this trace exercises:
#   unified_attention_d<D>_<dtype>_<n?>mask_nopage.cpp
MASKTAG=$([[ "$MASK" == "0" ]] && echo "nmask" || echo "mask")
TRACE_INSTANCE="${TRACE_INSTANCE:-d${D}_${DTYPE}_${MASKTAG}_nopage}"

if [[ "${LINETABLES:-0}" == "1" ]]; then
    if [[ "$SLIM" == "1" ]]; then
        echo "[run] (1/3) SLIM rebuild: only instance '$TRACE_INSTANCE' + line tables ..."
        SLIM_ENV=(AITER_UA_TRACE_INSTANCES="$TRACE_INSTANCE")
        REBUILD=1   # wipe build dir for a clean full<->slim switch (only 4 TUs)
    else
        echo "[run] (1/3) FULL rebuild with -gline-tables-only ..."
        SLIM_ENV=()
        REBUILD=2
    fi
    ( cd "$AITER_ROOT" && env "${SLIM_ENV[@]}" \
        AITER_REBUILD="$REBUILD" AITER_EXTRA_HIP_FLAGS="-gline-tables-only" \
        HIP_VISIBLE_DEVICES="$GPU" python3 op_tests/test_unified_attention_ck.py \
        -b "$BATCH" -sq "$SQ" -sk "$SK" --num-heads 16,2 --head-size "$D" --block-size 64 \
        --dtype "$DTYPE" --num-blocks auto --mask-type "$MASK" --contiguous \
        --no-triton --no-reference --seed 42 \
        > "$SCRIPTS/linetables_build.log" 2>&1 )
    echo "[run]     build done (log: linetables_build.log)"
else
    echo "[run] (1/3) skipping rebuild (set LINETABLES=1 to embed source / switch SLIM)"
fi

echo "[run] (2/3) collecting ATT trace -> $TAG ..."
TAG="$TAG" GPU="$GPU" bash "$SCRIPTS/rocprof_att_prefill.sh" "$DTYPE" "$BATCH" "$SQ" "$SK"

echo "[run] (3/3) building overlap report (SIMD $SIMD, $ITERS iters) ..."
cd "$SCRIPTS"
python3 -m att_analysis.report "$RUN_DIR" --simd "$SIMD" --iters "$ITERS"
echo "[run] done -> $RUN_DIR/att_analysis/"
