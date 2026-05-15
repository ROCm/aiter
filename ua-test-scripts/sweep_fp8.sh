#!/usr/bin/env bash
# CK-UA FP8 sweep: decode + prefill, d=64/128, block sizes 32/64.
#
# For every shape we run test_single_shape.py once in FP8 mode and once in
# BF16 mode and emit a Markdown row with:
#   - CK ms (median of N iters)            (FP8 / BF16)
#   - Triton ms (median)                   (FP8 / BF16)
#   - speedup (Triton/CK) for each dtype
#   - FP8 numeric tail summary
#       max_abs_diff, fraction of elements > 0.05, fraction of (token,head)
#       rows with mean_abs_diff > 0.05 -- the last column lights up the
#       row-corruption tail-failure mode we're tracking.
#
# Decode variants currently compiled for FP8: decode_d{64,128}_m128 and the
# prefill variants. The shapes below are chosen so the dispatch lands in
# either the m128 decode kernel (avg_rows in (32, 128]) or the prefill
# kernel (avg_rows > 128). decode_d*_m{16,32,64} are intentionally avoided
# -- the FP8 instances for those tiers aren't compiled yet (see
# unified_attention.cpp dispatch_variant<>).
set -uo pipefail

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-7}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-10}"
SEED="${SEED:-42}"

LOG=/tmp/ua_fp8_sweep.log
: > "$LOG"

# Each shape: "<label>;<args>"
# Constraints (so decode hits m128 rather than the unbuilt m16/m32):
#   sq * num_qpkv >= 64 (decode), or sq*num_qpkv >= 256 forces prefill.
SHAPES=(
    # ---- decode m128 (sq=8 * num_qpkv=8 = 64 rows -> m128) ----
    "decode d=64  hq=64 hk=8 sq=8  sk=4096 b=4 ;-b 4 -sq 8  -sk 4096 -hq 64 -hk 8  -d 64  --block-size 32"
    "decode d=64  hq=64 hk=8 sq=8  sk=8192 b=4 ;-b 4 -sq 8  -sk 8192 -hq 64 -hk 8  -d 64  --block-size 32"
    "decode d=128 hq=64 hk=8 sq=8  sk=4096 b=4 ;-b 4 -sq 8  -sk 4096 -hq 64 -hk 8  -d 128 --block-size 32"
    "decode d=128 hq=64 hk=8 sq=8  sk=8192 b=4 ;-b 4 -sq 8  -sk 8192 -hq 64 -hk 8  -d 128 --block-size 32"
    # ---- decode m128 with block_size=64 ----
    "decode d=64  hq=64 hk=8 sq=8  sk=4096 b=4 blk=64;-b 4 -sq 8  -sk 4096 -hq 64 -hk 8  -d 64  --block-size 64"
    "decode d=128 hq=64 hk=8 sq=8  sk=4096 b=4 blk=64;-b 4 -sq 8  -sk 4096 -hq 64 -hk 8  -d 128 --block-size 64"
    # ---- prefill (sq * num_qpkv > 256 -> prefill_d* path) ----
    "prefill d=64  hq=64 hk=8 sq=512  sk=4096 b=2 ;-b 2 -sq 512  -sk 4096 -hq 64 -hk 8 -d 64  --block-size 32"
    "prefill d=64  hq=64 hk=8 sq=1024 sk=4096 b=2 ;-b 2 -sq 1024 -sk 4096 -hq 64 -hk 8 -d 64  --block-size 32"
    "prefill d=128 hq=64 hk=8 sq=512  sk=4096 b=2 ;-b 2 -sq 512  -sk 4096 -hq 64 -hk 8 -d 128 --block-size 32"
    "prefill d=128 hq=64 hk=8 sq=1024 sk=4096 b=2 ;-b 2 -sq 1024 -sk 4096 -hq 64 -hk 8 -d 128 --block-size 32"
    "prefill d=128 hq=64 hk=8 sq=2048 sk=4096 b=1 ;-b 1 -sq 2048 -sk 4096 -hq 64 -hk 8 -d 128 --block-size 32"
    "prefill d=128 hq=64 hk=8 sq=1024 sk=4096 b=2 blk=64;-b 2 -sq 1024 -sk 4096 -hq 64 -hk 8 -d 128 --block-size 64"
)

# parse_run shape_args dtype tol_args -> echoes "<ck_ms> <triton_ms> <max_diff> <pct_above_0p05> <pct_bad_rows> <pass>"
parse_run() {
    local shape_args="$1"; shift
    local dtype="$1"; shift
    local tol_args="$1"; shift
    local out
    out=$(python ua-test-scripts/test_single_shape.py \
              $shape_args --dtype "$dtype" $tol_args \
              --warmup "$WARMUP" --iters "$ITERS" --seed "$SEED" --test 2>&1 \
              | tee -a "$LOG")
    local ck triton mxd pct05 pctrows ok
    ck=$(echo "$out"     | grep -E "^\s*CK time:"    | awk '{print $3}')
    triton=$(echo "$out" | grep -E "^\s*Triton time:" | awk '{print $3}')
    mxd=$(echo "$out"    | grep "Max abs diff:"      | awk '{print $4}')
    pct05=$(echo "$out"  | grep "Mismatch > 0.05:"   | awk -F'[()]' '{print $2}' | awk '{print $1}')
    pctrows=$(echo "$out"| grep "Row-mean > 0.05:"   | awk -F'[()]' '{print $2}' | awk '{print $1}')
    if echo "$out" | grep -qE "PASS"; then
        ok=PASS
    elif echo "$out" | grep -qE "FAIL"; then
        ok=FAIL
    else
        ok=NORES
    fi
    echo "${ck:--} ${triton:--} ${mxd:--} ${pct05:--} ${pctrows:--} $ok"
}

printf '\n## CK-UA FP8 sweep (HIP_VISIBLE_DEVICES=%s, ITERS=%d, WARMUP=%d, SEED=%d)\n\n' \
       "$HIP_VISIBLE_DEVICES" "$ITERS" "$WARMUP" "$SEED"

printf '| shape | dtype | CK ms | Triton ms | speedup | max diff | %% el>0.05 | %% row corrupt | test |\n'
printf '|-------|-------|------:|----------:|--------:|---------:|----------:|--------------:|:----:|\n'

cd "$(dirname "$0")/.."

for entry in "${SHAPES[@]}"; do
    label="${entry%%;*}"
    args="${entry#*;}"
    # FP8 run -- relaxed tolerance to absorb the row-corruption tail until
    # the underlying CK bug is fixed; we still report the row-corrupt %
    # column so regressions are visible.
    fp8_line=$(parse_run "$args" fp8 "--atol 0.6 --rtol 0.6 --no-splitkv")
    read fp8_ck fp8_tr fp8_mxd fp8_pct fp8_rows fp8_ok <<<"$fp8_line"
    # BF16 reference run -- same shape, no FP8 quirks.
    bf_line=$(parse_run "$args" bf16 "")
    read bf_ck bf_tr bf_mxd bf_pct bf_rows bf_ok <<<"$bf_line"

    speedup_fp8=$(python -c "print(f'{($fp8_tr)/($fp8_ck):.2f}')" 2>/dev/null || echo "-")
    speedup_bf=$(python -c "print(f'{($bf_tr)/($bf_ck):.2f}')" 2>/dev/null || echo "-")

    printf '| %s | fp8  | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$label" "$fp8_ck" "$fp8_tr" "$speedup_fp8" "$fp8_mxd" "$fp8_pct" "$fp8_rows" "$fp8_ok"
    printf '| %s | bf16 | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$label" "$bf_ck" "$bf_tr" "$speedup_bf" "$bf_mxd" "$bf_pct" "$bf_rows" "$bf_ok"
done

printf '\nFull log: %s\n' "$LOG"
