#!/usr/bin/env bash
# CK-UA FP8-only sweep — same shapes as sweep_fp8.sh but skips bf16
# to halve the wall time. Use when you only need fp8 numbers.
set -uo pipefail

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-7}"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-10}"
SEED="${SEED:-42}"

LOG=/tmp/ua_fp8_only_sweep.log
: > "$LOG"

SHAPES=(
    # ---- decode high-batch (saturates 2D grid -> no split-KV on either side) ----
    "decode d=64  sq=1 sk=128000 b=128 ;-b 128 -sq 1 -sk 128000 -hq 64 -hk 8 -d 64  --block-size 32"
    "decode d=64  sq=1 sk=128000 b=256 ;-b 256 -sq 1 -sk 128000 -hq 64 -hk 8 -d 64  --block-size 32"
    "decode d=128 sq=1 sk=128000 b=128 ;-b 128 -sq 1 -sk 128000 -hq 64 -hk 8 -d 128 --block-size 32"
    "decode d=128 sq=1 sk=128000 b=256 ;-b 256 -sq 1 -sk 128000 -hq 64 -hk 8 -d 128 --block-size 32"
    # ---- decode low-batch (split-KV path is on for BOTH backends) ----
    "decode-lo d=64  sq=1 sk=128000 b=4 ;-b 4 -sq 1 -sk 128000 -hq 64 -hk 8 -d 64  --block-size 32"
    "decode-lo d=128 sq=1 sk=128000 b=4 ;-b 4 -sq 1 -sk 128000 -hq 64 -hk 8 -d 128 --block-size 32"
    # ---- prefill (sq=sk=75600, b=1) -> prefill_d{64,128} (8 warps) ----
    "prefill d=64  sq=75600 sk=75600 b=1 ;-b 1 -sq 75600 -sk 75600 -hq 64 -hk 8 -d 64  --block-size 32"
    "prefill d=128 sq=75600 sk=75600 b=1 ;-b 1 -sq 75600 -sk 75600 -hq 64 -hk 8 -d 128 --block-size 32"
)

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

printf '\n## CK-UA FP8-only sweep (HIP_VISIBLE_DEVICES=%s, ITERS=%d, WARMUP=%d, SEED=%d)\n\n' \
       "$HIP_VISIBLE_DEVICES" "$ITERS" "$WARMUP" "$SEED"

printf '| shape | CK ms | Triton ms | speedup | max diff | %% el>0.05 | %% row corrupt | test |\n'
printf '|-------|------:|----------:|--------:|---------:|----------:|--------------:|:----:|\n'

cd "$(dirname "$0")/.."

for entry in "${SHAPES[@]}"; do
    label="${entry%%;*}"
    args="${entry#*;}"
    fp8_line=$(parse_run "$args" fp8 "--atol 0.6 --rtol 0.6")
    read fp8_ck fp8_tr fp8_mxd fp8_pct fp8_rows fp8_ok <<<"$fp8_line"
    speedup_fp8=$(python -c "print(f'{($fp8_tr)/($fp8_ck):.2f}')" 2>/dev/null || echo "-")
    printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$label" "$fp8_ck" "$fp8_tr" "$speedup_fp8" "$fp8_mxd" "$fp8_pct" "$fp8_rows" "$fp8_ok"
done

printf '\nFull log: %s\n' "$LOG"
