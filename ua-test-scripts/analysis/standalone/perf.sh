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

# Decode auto-config. The decode tiers are paged-only instances selected at
# runtime by rows = sq * (hq/hk): <=16 -> m16 (decode_t), <=32 -> m32
# (decode_s), <=128 -> m128 (decode). When SK is set (sq=1 decode), pick the
# matching target instance + force PAGED=1 unless the caller overrode them, so
# `SK=196608 perf.sh 1 16 2 128 0` just works. PAGE_BLK defaults to 64.
MASKTAG=$([[ "$MASK" == "0" ]] && echo "nmask" || echo "mask")
if [[ -n "${SK:-}" ]]; then
    rows=$(( SQ * HQ / HK ))
    if   [[ $rows -le 16  ]]; then tier="decode_t"
    elif [[ $rows -le 32  ]]; then tier="decode_s"
    elif [[ $rows -le 128 ]]; then tier="decode"
    else tier=""; fi   # rows>128 isn't a decode tier; leave default (prefill)
    [[ -n "$tier" ]] && export TARGET_INSTANCE="${TARGET_INSTANCE:-unified_attention_d${D}_${DTYPE}_${MASKTAG}_${tier}}"
    export PAGED="${PAGED:-1}"; export PAGE_BLK="${PAGE_BLK:-64}"
fi

# build.sh self-guards (rebuilds iff stamp mismatch or any source newer than exe).
ARCH="$ARCH" DTYPE="$DTYPE" D="$D" MASK="$MASK" bash "$HERE/build.sh" >&2

run() {  # $1=sq
    local env=(HIP_VISIBLE_DEVICES="$GPU" PERF=1 WARMUP="$WARMUP" ROTATE="$ROTATE")
    [[ -n "${SK:-}"         ]] && env+=(SK="$SK")
    [[ -n "${PAGED:-}"      ]] && env+=(PAGED="$PAGED")
    [[ -n "${PAGE_BLK:-}"   ]] && env+=(PAGE_BLK="$PAGE_BLK")
    [[ -n "${NUM_SPLITS:-}" ]] && env+=(NUM_SPLITS="$NUM_SPLITS")
    [[ -n "${NUM_SEQS:-}"   ]] && env+=(NUM_SEQS="$NUM_SEQS")
    env "${env[@]}" "$EXE" "$1" "$HQ" "$HK" "$D" "$MASK" "$ITERS" | sed -n 's/^\[perf\]/  /p'
}

echo "# UA standalone perf  dtype=$DTYPE d=$D mask=$MASK heads=$HQ/$HK  iters=$ITERS rotate=$ROTATE${SK:+ sk=$SK paged=$PAGE_BLK target=${TARGET_INSTANCE:-?}}"
if [[ -n "${SWEEP:-}" ]]; then
    for s in $SWEEP; do printf 'sq=%-7s' "$s"; run "$s"; done
else
    printf 'sq=%-7s' "$SQ"; run "$SQ"
fi
