#!/usr/bin/env bash
# Build + run the standalone accuracy check: compares the kernel output against
# a host float reference (O = softmax(scale*Q.K^T [+causal]).V) computed from the
# SAME fp8 bytes the kernel reads. This validates that the trace bench's setup
# (layouts / strides / fp8 e4m3 encoding / GQA head mapping / mask) is faithful,
# so profiling results aren't confounded by a broken harness.
#
# Tolerances + pass rule mirror op_tests/test_unified_attention_ck.py's
# checkAllclose for fp8: atol=rtol=0.15, PASS if <5% of elements exceed it.
#
#   standalone/check.sh [sq] [hq] [hk] [d] [mask]     (default 512 16 2 128 0)
#
# Keep sq small -- the host reference is O(hq * sq^2 * d) on the CPU.
# Env: GPU=2  ARCH=gfx950  DTYPE=fp8  SEED=42  QKV_STD=1.0  RTOL/ATOL/TOL_ERR_RATIO
set -euo pipefail

SQ="${1:-512}"; HQ="${2:-16}"; HK="${3:-2}"; D="${4:-128}"; MASK="${5:-0}"
DTYPE="${DTYPE:-fp8}"; GPU="${GPU:-2}"; ARCH="${ARCH:-gfx950}"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXE="$HERE/build/ua_trace"

# build.sh self-guards (rebuilds iff stamp mismatch or any source newer than exe).
ARCH="$ARCH" DTYPE="$DTYPE" D="$D" MASK="$MASK" bash "$HERE/build.sh" >&2

HIP_VISIBLE_DEVICES="$GPU" CHECK=1 SEED="${SEED:-42}" "$EXE" "$SQ" "$HQ" "$HK" "$D" "$MASK" 1
