#!/usr/bin/env bash
# Head-to-head gluon vs flydsl on the DSV4 fp8 linear shapes.
#
# One process per (M, backend-set) rather than one per shape: the harness times
# every backend in the same process, interleaved, min-of-trials. Splitting the
# backends across processes -- what this script used to do -- let clock drift
# and graph-pool fragmentation land entirely on whichever block ran second,
# which moved single shapes by 2-3x and inverted the gluon/flydsl verdict.
#
# flydsl is the incumbent, so it is listed first and every other backend is
# scored against it: "vs flydsl" > 1 means gluon is ahead, and "gap us" is what
# gluon still owes.
#
# Env overrides:
#   BACKENDS=flydsl,gluon,flydsl_raw   first is the baseline; flydsl_raw skips
#                                      the aiter dispatch layer
#   METHODS=event,cudagraph,perftest   first is the headline; perftest also
#                                      prints the per-kernel device-time split
#   TRIALS=5                           interleaved repeats (min is reported)

set -euo pipefail

export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=1
export HSA_USE_SVM=1
export HSA_XNACK=1
export ENABLE_CK=0

export TRITON_CACHE_DIR=$(realpath ./)/cache_mxfp8_gemm
rm -rf "$TRITON_CACHE_DIR"

BACKENDS=${BACKENDS:-flydsl,gluon}
METHODS=${METHODS:-event}
TRIALS=${TRIALS:-5}

run_sweep() {
    local m=$1 sweep=$2
    echo "############ M=${m}  (${sweep}) ############"
    python bench_gemm_afp8wfp8_preshuffle.py \
        --m "${m}" --sweep "${sweep}" \
        --transpose-x-scale \
        --backend "${BACKENDS}" \
        --method "${METHODS}" \
        --trials "${TRIALS}"
}

# sweep for TP4
# run_sweep 16 tp4
# run_sweep 64 tp4
# run_sweep 16384 tp4

# sweep for EP4 
# run_sweep 512 ep4
# run_sweep 16384 ep4





# for individual shape:

BACKENDS=flydsl,gluon
METHODS=event # perftest is busted on b45-1
TRIALS=5

python bench_gemm_afp8wfp8_preshuffle.py \
        --m $1 --n $2 --k $3 \
        --transpose-x-scale \
        --backend "${BACKENDS}" \
        --method "${METHODS}" \
        --trials "${TRIALS}"
    
grep -rnw ".globl" $TRITON_CACHE_DIR
grep -rnw ".vgpr_count" $TRITON_CACHE_DIR
grep -rnw ".vgpr_spill_count" $TRITON_CACHE_DIR