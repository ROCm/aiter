#!/usr/bin/env bash
# Simple perf harness for flydsl_qk_norm_rope_quant -- prefill point.
#   bs=16, tok=16384  -> T=16384 flat tokens  (this perf path is token-flat:
#                        q is [T, H*D]; there is no separate batch dim, so
#                        bs*seqlen collapses to the total token count T).
#   H=128, D=512, RD=64, BF16 (no-quant).
#
# By default sweeps BOTH q_weight variants (--qweight makes the test emit a
# qw=False row and a qw=True row), so you get a direct with/without comparison:
#     q_weight off -> qk_norm_rope_H128_D512_RD64_r32_w32_flydsl
#     q_weight on  -> qk_norm_rope_H128_D512_RD64_qw_r32_w32_flydsl
# (Set QWEIGHT=0 to run the no-q_weight row only.)
#
# At these shapes the public API selects TDM CT=16, which reuses cos/sin across
# all 16 eight-row tiles (one complete H=128 token) in each workgroup.
#
# Usage:
#   ./runperf-qknorm-bs16-t16384.sh          # sweep qw off+on -> us / GB/s / %peak
#   QWEIGHT=0 ./runperf-qknorm-bs16-t16384.sh # no-q_weight only
#   TRACE=1 ./runperf-qknorm-bs16-t16384.sh  # + kernel-trace to confirm r32_w32
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
set -euo pipefail

TEST="${TEST:-op_tests/test_flydsl_qk_norm_rope_quant.py}"
H="${H:-128}"
D="${D:-512}"
RD="${RD:-64}"
T="${T:-16384}"                  # bs(16) * seqlen(1024) = 16384 total tokens
QWEIGHT="${QWEIGHT:-1}"          # 1 -> also run q_weight=enabled row
TRACE="${TRACE:-0}"
OUTDIR="${OUTDIR:-/tmp/prof_qknorm_bs16_t16384}"

COMMON="--H ${H} --D ${D} --RD ${RD} --no-quant"
[[ "$QWEIGHT" == "1" ]] && COMMON="${COMMON} --qweight"
# Expected kernel symbol(s) at this shape/path (r32_w32 = ROWS_PER_WG=32, wave32).
# qw variant carries an extra "qw" tag; regex matches both.
KREGEX="${KREGEX:-qk_norm_rope_tdm_H${H}_D${D}_RD${RD}_.*ct16.*flydsl}"

echo "config: bs=16 tok=${T} (T=${T})  H=${H} D=${D} RD=${RD} dtype=BF16  QWEIGHT=${QWEIGHT}  TRACE=${TRACE}"
echo "expect kernel(s): qk_norm_rope_tdm_H${H}_D${D}_RD${RD}_*_ct16[_h][_qw]_fused_w32_flydsl"
echo

# --- perf sweep (qw off + on) ------------------------------------------------
python "$TEST" -T ${T} $COMMON

# --- also run T=512 if the primary T is not already 512 ----------------------
if [[ "$T" != "512" ]]; then
    echo
    echo "=== T=512 sweep ==="
    python "$TEST" -T 512 $COMMON
fi

# --- optional: confirm the real dispatched kernel name(s) are TDM ct16 -------
if [[ "$TRACE" == "1" ]]; then
    echo
    echo "=== kernel-trace: confirm TDM ct16 dispatch ==="
    rm -rf "$OUTDIR"
    rocprofv3 --kernel-trace -d "$OUTDIR" \
        -- python "$TEST" -T ${T} $COMMON >/dev/null 2>&1
    if grep -rh qk_norm_rope "$OUTDIR"/ 2>/dev/null | grep -oE "$KREGEX" | sort -u; then
        :
    else
        echo "WARNING: no match for '$KREGEX' -- inspect $OUTDIR/ manually"
    fi
fi
