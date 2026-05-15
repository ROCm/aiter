#!/usr/bin/env bash
# CK-UA: small-cache (buffer_load) vs big-cache (global_load_lds) sweep.
#
# For each shape we run test_single_shape.py twice — once with the default
# (small) num_blocks so cache_ptr_int32_overflow_possible == false (the
# original `async_load_tile_raw` path), and once with --num-blocks
# NUM_BLOCKS_BIG so the cache exceeds INT32_MAX elements and the kernel
# falls back to `async_load_tile_raw_long`. Three iterations per
# configuration; we keep the median CK ms.
#
# Output: a single Markdown table with small-cache vs big-cache numbers,
# Triton numbers, and the slowdown / Triton-gap-change for each row.
set -euo pipefail

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-7}"
SK=128000
NUM_BLOCKS_BIG=200000
ITERS_PER_CFG=3

# Shape list: each entry is "<label>;<args-to-test_single_shape.py-w/o-sk-and-num-blocks>"
SHAPES=(
    # ---- decode (sq=1) ----
    "decode d=64  hq=8   hk=1  b=32; -b 32 -sq 1 -hq 8   -hk 1  -d 64"
    "decode d=64  hq=64  hk=8  b=32; -b 32 -sq 1 -hq 64  -hk 8  -d 64"
    "decode d=64  hq=128 hk=16 b=32; -b 32 -sq 1 -hq 128 -hk 16 -d 64"
    "decode d=128 hq=8   hk=1  b=32; -b 32 -sq 1 -hq 8   -hk 1  -d 128"
    "decode d=128 hq=64  hk=8  b=32; -b 32 -sq 1 -hq 64  -hk 8  -d 128"
    "decode d=128 hq=128 hk=16 b=32; -b 32 -sq 1 -hq 128 -hk 16 -d 128"
    "decode d=64  hq=64  hk=8  b=128;-b 128 -sq 1 -hq 64  -hk 8  -d 64"
    "decode d=128 hq=64  hk=8  b=128;-b 128 -sq 1 -hq 64  -hk 8  -d 128"
    # ---- prefill (large sq, b=1 to keep memory in check) ----
    "prefill d=64  hq=8 hk=8 sq=2048 b=1; -b 1 -sq 2048 -hq 8 -hk 8 -d 64"
    "prefill d=64  hq=8 hk=8 sq=4096 b=1; -b 1 -sq 4096 -hq 8 -hk 8 -d 64"
    "prefill d=128 hq=8 hk=8 sq=2048 b=1; -b 1 -sq 2048 -hq 8 -hk 8 -d 128"
    "prefill d=128 hq=8 hk=8 sq=4096 b=1; -b 1 -sq 4096 -hq 8 -hk 8 -d 128"
)

LOG=/tmp/ua_sweep.log
: > "$LOG"

# parse_run shape_args extra_args -> echoes "<ck_ms> <triton_ms> <pass_or_fail>"
parse_run() {
    local shape_args="$1"; shift
    local extra_args="$1"; shift
    local out
    out=$(python ua-test-scripts/test_single_shape.py \
              $shape_args -sk "$SK" $extra_args --test 2>&1 | tee -a "$LOG")
    local ck triton ok
    ck=$(echo "$out" | grep -E "^\s*CK time:" | awk '{print $3}')
    triton=$(echo "$out" | grep -E "^\s*Triton time:" | awk '{print $3}')
    if echo "$out" | grep -q "Correctness: ✓ PASS"; then
        ok=PASS
    elif echo "$out" | grep -q "FAIL"; then
        ok=FAIL
    else
        ok=NO_RESULT
    fi
    echo "$ck $triton $ok"
}

# median3 a b c -> echoes middle value
median3() {
    printf "%s\n%s\n%s\n" "$1" "$2" "$3" | sort -g | sed -n 2p
}

printf "Sweep on %s (HIP_VISIBLE_DEVICES=%s), sk=%d, big-cache num_blocks=%d\n" \
       "$(rocminfo 2>/dev/null | grep gfx | head -1 | awk '{print $2}')" \
       "$HIP_VISIBLE_DEVICES" "$SK" "$NUM_BLOCKS_BIG"
echo

# Print Markdown table header.
printf "| %-38s | %9s %9s %8s | %9s %9s %8s | %9s | %9s %9s |\n" \
    "shape" "CK-small" "Tri-small" "vs-Tri" \
            "CK-big"   "Tri-big"   "vs-Tri" \
            "CK-slow" "gap-small" "gap-big"
printf "| %-38s | %9s %9s %8s | %9s %9s %8s | %9s | %9s %9s |\n" \
    "$(printf '%.0s-' {1..38})" "---ms" "---ms" "---x" "---ms" "---ms" "---x" "small/big" "vs-Tri" "vs-Tri"

for entry in "${SHAPES[@]}"; do
    label="${entry%%;*}"
    args="${entry#*;}"

    declare -a ck_s=() tri_s=() ck_b=() tri_b=()
    ok_s=PASS; ok_b=PASS
    for i in $(seq 1 "$ITERS_PER_CFG"); do
        # small cache (default num_blocks)
        read -r ck tri ok < <(parse_run "$args" "")
        ck_s+=("$ck"); tri_s+=("$tri")
        [ "$ok" = "PASS" ] || ok_s=$ok
        # big cache (overflow flag → long path)
        read -r ck tri ok < <(parse_run "$args" "--num-blocks $NUM_BLOCKS_BIG")
        ck_b+=("$ck"); tri_b+=("$tri")
        [ "$ok" = "PASS" ] || ok_b=$ok
    done

    ck_s_med=$(median3 "${ck_s[@]}")
    tri_s_med=$(median3 "${tri_s[@]}")
    ck_b_med=$(median3 "${ck_b[@]}")
    tri_b_med=$(median3 "${tri_b[@]}")

    # vs-Triton (>1 means CK wins, <1 means Triton wins)
    gap_s=$(python -c "print(f'{$tri_s_med / $ck_s_med:.3f}')")
    gap_b=$(python -c "print(f'{$tri_b_med / $ck_b_med:.3f}')")
    slowdown=$(python -c "print(f'{$ck_b_med / $ck_s_med:.3f}')")

    ok_tag=""
    [ "$ok_s" = "PASS" ] || ok_tag+=" [small:$ok_s]"
    [ "$ok_b" = "PASS" ] || ok_tag+=" [big:$ok_b]"

    printf "| %-38s | %9.3f %9.3f %7sx | %9.3f %9.3f %7sx | %8sx | %8sx %8sx |%s\n" \
        "$label" \
        "$ck_s_med" "$tri_s_med" "$gap_s" \
        "$ck_b_med" "$tri_b_med" "$gap_b" \
        "$slowdown" "$gap_s" "$gap_b" \
        "$ok_tag"
done

echo
echo "Per-iteration log: $LOG"
