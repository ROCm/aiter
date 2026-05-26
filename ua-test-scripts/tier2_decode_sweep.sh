#!/usr/bin/env bash
# Decode perf sweep across batch / seqlen / dtype to evaluate the Tier-2
# LDS page-table cache enablement for TinyDecode (kBlockSize == 64).
#
# Outputs CK ms + Triton ms (and ratio) for each (b, sk, dt) combo. Run
# this before and after toggling the gate to compare.
set -euo pipefail
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-7}"
cd "$(dirname "$0")/.."

WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"

# (b, hq, hk, d) shapes — small-b MHA hits decode_d128_m16/TinyDecode;
# larger-b GQA hits decode_d128_m128 (8-warp prefill-style). Both are
# affected by the gate change, so we measure both.
#
# format: label batch sq sk hq hk d
declare -a CASES=(
    "b1_GQA8       1   1  4096  64  8  128"
    "b1_GQA8       1   1  16384 64  8  128"
    "b1_GQA8       1   1  65536 64  8  128"
    "b1_GQA8       1   1  131072 64 8  128"
    "b1_MHA        1   1  4096   4  4 128"
    "b1_MHA        1   1  16384  4  4 128"
    "b1_MHA        1   1  65536  4  4 128"
    "b1_MHA        1   1  131072 4  4 128"
    "b32_GQA4      32  1  4096  32  8 128"
    "b32_GQA4      32  1  16384 32  8 128"
    "b32_GQA4      32  1  65536 32  8 128"
    "b128_GQA4    128  1  4096  32  8 128"
    "b128_GQA4    128  1  16384 32  8 128"
    "b128_GQA4    128  1  65536 32  8 128"
    "b256_GQA4    256  1  4096  32  8 128"
    "b256_GQA4    256  1  16384 32  8 128"
    "b256_GQA4    256  1  65536 32  8 128"
)

printf '  %-14s %3s  %-7s %3s %3s %4s  %-5s %10s %10s %7s\n' \
    "label" "b" "sk" "hq" "hk" "d" "dt" "CK_ms" "Triton_ms" "CK/Tri"
echo "  $(printf '%.0s-' {1..96})"

for entry in "${CASES[@]}"; do
    read label b sq sk hq hk d <<< "$entry"
    for dt in bf16 fp8; do
        out=$(python3 ua-test-scripts/test_single_shape.py \
              -b "$b" -sq "$sq" -sk "$sk" -hq "$hq" -hk "$hk" -d "$d" \
              --dtype "$dt" --warmup "$WARMUP" --iters "$ITERS" --seed 42 2>/dev/null || true)
        ck=$(echo "$out" | grep -E "^\s*CK time:" | awk '{print $3}')
        tr=$(echo "$out" | grep -E "^\s*Triton time:" | awk '{print $3}')
        if [[ -n "${ck:-}" && -n "${tr:-}" ]]; then
            ratio=$(awk "BEGIN { printf \"%.3f\", $ck / $tr }")
        else
            ratio="—"
        fi
        printf '  %-14s %3d  %-7d %3d %3d %4d  %-5s %10s %10s %7s\n' \
            "$label" "$b" "$sk" "$hq" "$hk" "$d" "$dt" "${ck:-—}" "${tr:-—}" "$ratio"
    done
done
