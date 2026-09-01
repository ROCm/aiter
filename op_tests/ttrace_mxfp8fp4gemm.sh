#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# rocprofv3 --att (thread-trace) driver for the gfx1250 mxfp8fp4 F8GEMM UT.
#
# Why a wrapper: rocprofv3 is a process-level tool -- it intercepts the HSA queue
# to inject the SQTT thread trace, and that CANNOT coexist with the UT's internal
# torch.profiler (event/API-trace based; also owns the queue). So rocprofv3 wraps
# the whole process from the OUTSIDE, and the UT runs `--mode ttrace`, which turns
# the internal profiler off and just fires a clean back-to-back dispatch loop
# (aiter.test_common.run_dispatch_loop). --kernel-iteration-range then picks one
# steady-state dispatch out of that loop to trace.
#
# Usage:
#   ./ttrace_mxfp8fp4gemm.sh
#   INTYPE=a8w4 SHAPE=2,1048576,16384 ITER=50 ./ttrace_mxfp8fp4gemm.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PY:-python3}"
INTYPE="${INTYPE:-a8w8}"
SHAPE="${SHAPE:-2,1048576,16384}"   # memory-bound decode representative
# Kernel-name filter for the trace. The heuristic dispatch emits an f8gemm .co;
# a bare "f8gemm" matches it (the golden is skipped in ttrace mode, so nothing
# else runs). Narrow it to the exact mangled name via KNL_REGEX if needed.
KNL_REGEX="${KNL_REGEX:-f8gemm}"
ITER="${ITER:-50}"                  # which dispatch to trace (0-based, steady state)
PROF_DIR="${PROF_DIR:-./att_out}"

set -x
rocprofv3 --att \
    --att-simd-select 0 \
    --att-target-cu 1 \
    --att-shader-engine-mask 0x1 \
    --kernel-trace \
    --kernel-include-regex "${KNL_REGEX}" \
    --kernel-iteration-range "[${ITER}]" \
    --truncate-kernels \
    -d "${PROF_DIR}" -o att \
    -- "${PY}" test_mxfp8fp4gemm.py --mode ttrace --intype "${INTYPE}" -s "${SHAPE}"
