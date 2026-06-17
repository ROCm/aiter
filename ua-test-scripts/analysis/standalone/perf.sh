#!/usr/bin/env bash
# Standalone perf measurement for the CK unified-attention kernel -- a JIT-free
# replacement for the Python perf harness. It calls the SAME ck_tile kernel the
# aiter op dispatches to, but compiled fresh from source and STAMPED per build
# (build/.built), so there is never any "is the JIT module stale?" ambiguity.
#
# The kernel INSTANCE is selected at compile time only by (dtype, d, mask); the
# shape (sq/hq/hk) is a runtime arg -- so one build sweeps all sequence lengths,
# and only changing dtype/d/mask triggers a rebuild (handled automatically).
#
#   standalone/perf.sh [sq] [hq] [hk] [d] [mask]        (single shape)
#   SWEEP="1024 2048 4096 8192 16384" standalone/perf.sh   (sweep sq)
#
# Env: GPU=2 ARCH=gfx950 DTYPE=fp8  ITERS=100 WARMUP=10 ROTATE=1
#   ROTATE>1 allocates N independent Q/K/V copies to defeat L2 reuse (use for
#   small shapes that fit in L2; large prefill shapes are fine at 1).
set -euo pipefail

SQ="${1:-8192}"; HQ="${2:-16}"; HK="${3:-2}"; D="${4:-128}"; MASK="${5:-0}"
DTYPE="${DTYPE:-fp8}"; GPU="${GPU:-2}"; ARCH="${ARCH:-gfx950}"
ITERS="${ITERS:-100}"; WARMUP="${WARMUP:-10}"; ROTATE="${ROTATE:-1}"

HERE="$(cd "$(dirname "$0")" && pwd)"
EXE="$HERE/build/ua_trace"

# build.sh self-guards (rebuilds iff stamp mismatch or any source newer than exe).
ARCH="$ARCH" DTYPE="$DTYPE" D="$D" MASK="$MASK" bash "$HERE/build.sh" >&2

run() {  # $1=sq
    HIP_VISIBLE_DEVICES="$GPU" PERF=1 WARMUP="$WARMUP" ROTATE="$ROTATE" \
        "$EXE" "$1" "$HQ" "$HK" "$D" "$MASK" "$ITERS" | sed -n 's/^\[perf\]/  /p'
}

echo "# UA standalone perf  dtype=$DTYPE d=$D mask=$MASK heads=$HQ/$HK  iters=$ITERS rotate=$ROTATE"
if [[ -n "${SWEEP:-}" ]]; then
    for s in $SWEEP; do printf 'sq=%-7s' "$s"; run "$s"; done
else
    printf 'sq=%-7s' "$SQ"; run "$SQ"
fi
