#!/bin/bash
# DSv4-Pro FlyDSL MoE ubench: TP + EP shapes, rand + const.
# From config.json: hidden 7168, moe_inter 3072, 384 experts, topk 6.
#   TP -> 384 experts local, inter 3072/TP     (grouped gemm bench)
#   EP -> 384/EP local experts, inter 3072     (test_moe_ep fake-EP)
# Both single-GPU; no all2all cost. Tuned rows exist only for TP=4 / EP=4.
#
# Usage: [TP=4] [EP=4] [TOKENS_TP=64] [TOKENS_EP=2048] [ITERS=128]
#        [EXPERTS_ACT=0] [LOG=./moe_ubench.log] op_tests/ubench_moe_dsv4_pro.sh

set -uo pipefail   # no -e: a missed accuracy gate must not kill later runs

AITER_DIR="${AITER_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # repo root = op_tests/..
TP="${TP:-4}"
EP="${EP:-4}"
TOKENS_TP="${TOKENS_TP:-64}"
TOKENS_EP="${TOKENS_EP:-2048}"
ITERS="${ITERS:-128}"
CONST_VAL="${CONST_VAL:-0.5}"
# Experts actually receiving routes. 0 = all (TP: 384, EP: 96 local).
# Valid range: [topk, experts]; TP also caps at tokens*topk.
EXPERTS_ACT="${EXPERTS_ACT:-0}"

INTER_TP=$((3072 / TP))
EXPERTS_EP=$((384 / EP))
# All run output is tee'd here, then parsed into the summary table at the end.
# Defaults to the invoking directory (captured before the cd below).
LOG="${LOG:-$PWD/moe_ubench.log}"
: > "$LOG"

cd "$AITER_DIR" || exit 1

export AITER_MOE_EXPERT_BALANCE=true   # TP bench only; test_moe_ep ignores it
export AITER_MOE_NUM_EXPERT_ACTIVATED="$EXPERTS_ACT"  # overrides EXPERT_BALANCE when >0
export AITER_LOG_MORE=1
export HIP_VISIBLE_DEVICES=0
export AITER_USE_GROUPED_GEMM=1        # redundant: auto-on for gfx1250
export AITER_FORCE_GFX1250=1           # redundant: no-op on a real gfx1250
export AITER_GROUPED_DEBUG=0           # redundant: default
export FLYDSL_DUMP_IR=0                # redundant: default
# AITER_GROUPED_GEMM_AS_PROLOGUE, AITER_GROUPED_GEMM_NAIVE,
# WEIGHT_SCALE_OP_SEL, --layout, --wst are deprecated

echo "== DSv4-Pro MoE ubench | TP=$TP EP=$EP iters=$ITERS | tokens TP=$TOKENS_TP EP=$TOKENS_EP | experts_act=$EXPERTS_ACT"

echo

echo "##### [1/8] e2e TP$TP 7168/$INTER_TP/384/topk6 a8w4 silu -- RAND #####" | tee -a "$LOG"
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench --data-format a8w4 --act silu \
  --model-dim 7168 --inter-dim "$INTER_TP" --experts 384 --topk 6 \
  --iters "$ITERS" --no-check-aot-cache --no-bias --tokens $TOKENS_TP | tee -a "$LOG"

echo
echo "##### [2/8] e2e TP$TP 7168/$INTER_TP/384/topk6 a8w4 silu -- CONST $CONST_VAL #####" | tee -a "$LOG"
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench --data-format a8w4 --act silu \
  --model-dim 7168 --inter-dim "$INTER_TP" --experts 384 --topk 6 \
  --iters "$ITERS" --no-check-aot-cache --no-bias --tokens $TOKENS_TP --const-init "$CONST_VAL" | tee -a "$LOG"

# -e is global experts, -ep divides it; -m is per-rank M in fake mode.
echo
echo "##### [3/8] e2e EP$EP 7168/3072/$EXPERTS_EP local/topk6 -- RAND #####" | tee -a "$LOG"
python3 -u op_tests/test_moe_ep.py \
  -t g1u1_a8w4_mxfp4 -hd 7168 -id 3072 -e 384 -k 6 -ep "$EP" \
  --ep-mode fake -m $TOKENS_EP | tee -a "$LOG"

echo
echo "##### [4/8] e2e EP$EP 7168/3072/$EXPERTS_EP local/topk6 -- CONST $CONST_VAL #####" | tee -a "$LOG"
python3 -u op_tests/test_moe_ep.py \
  -t g1u1_a8w4_mxfp4 -hd 7168 -id 3072 -e 384 -k 6 -ep "$EP" \
  --ep-mode fake -m $TOKENS_EP --const-init "$CONST_VAL" | tee -a "$LOG"

# Per-kernel MoE1/MoE2 timing (each looped alone). Separate runs because both
# harnesses report either e2e or per-kernel, not both: --scenario kernel for TP,
# AITER_EP_KERNEL_BENCH=1 for EP (which replaces the e2e us with gemm1+gemm2).
echo
echo "##### [5/8] kernel TP$TP MoE1/MoE2 -- RAND #####" | tee -a "$LOG"
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario kernel --data-format a8w4 --act silu \
  --model-dim 7168 --inter-dim "$INTER_TP" --experts 384 --topk 6 \
  --iters "$ITERS" --no-check-aot-cache --no-bias --tokens $TOKENS_TP | tee -a "$LOG"

echo
echo "##### [6/8] kernel TP$TP MoE1/MoE2 -- CONST $CONST_VAL #####" | tee -a "$LOG"
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario kernel --data-format a8w4 --act silu \
  --model-dim 7168 --inter-dim "$INTER_TP" --experts 384 --topk 6 \
  --iters "$ITERS" --no-check-aot-cache --no-bias --tokens $TOKENS_TP --const-init "$CONST_VAL" | tee -a "$LOG"

echo
echo "##### [7/8] kernel EP$EP MoE1/MoE2 -- RAND #####" | tee -a "$LOG"
AITER_EP_KERNEL_BENCH=1 python3 -u op_tests/test_moe_ep.py \
  -t g1u1_a8w4_mxfp4 -hd 7168 -id 3072 -e 384 -k 6 -ep "$EP" \
  --ep-mode fake -m $TOKENS_EP | tee -a "$LOG"

echo
echo "##### [8/8] kernel EP$EP MoE1/MoE2 -- CONST $CONST_VAL #####" | tee -a "$LOG"
AITER_EP_KERNEL_BENCH=1 python3 -u op_tests/test_moe_ep.py \
  -t g1u1_a8w4_mxfp4 -hd 7168 -id 3072 -e 384 -k 6 -ep "$EP" \
  --ep-mode fake -m $TOKENS_EP --const-init "$CONST_VAL" | tee -a "$LOG"

echo
echo "== done"

# Merge the 8 runs into one table: e2e us from runs 1-4, MoE1/MoE2 from 5-8.
# FLOP/byte model matches op_tests/bench_gfx1250_combo.py (g1u1: stage1 n=2*inter,
# a8w4 => 1 byte/act, 0.5 byte/weight, bf16 out).
python3 - "$LOG" "$INTER_TP" "$EXPERTS_EP" "$EXPERTS_ACT" <<'PYEOF'
import sys

log, inter_tp, experts_ep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
act = int(sys.argv[4])
MD, TOPK, AQ, WQ, BO = 7168, 6, 1.0, 0.5, 2
SHAPE = {"TP": (inter_tp, 384), "EP": (3072, experts_ep)}
data, mode, init, kind, hdr = {}, None, None, None, None

for ln in open(log, errors="ignore"):
    if ln.startswith("##### ["):
        f = ln.split()
        kind, mode = f[2], ("TP" if "TP" in f[3] else "EP")
        init = "const" if "CONST" in ln else "rand"
        hdr = None
    elif ln.startswith("|"):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if "tokens" in c or "global_token" in c:
            hdr = c
        elif hdr and len(c) == len(hdr) and not c[0].startswith(":"):
            d = dict(zip(hdr, c))
            try:
                tok = int(d.get("tokens") or d["global_token"])
            except (KeyError, ValueError):
                continue
            r = data.setdefault((mode, init, tok), {})
            if kind == "e2e":
                r["e2e"] = float(d["us"])
            else:
                for k in ("gemm1_us", "gemm2_us"):
                    if d.get(k):
                        r[k] = float(d[k])

cols = ("mode", "init", "tokens", "E_act", "e2e us", "MoE1 us", "MoE1 TFLOPS",
        "MoE1 TB/s", "MoE2 us", "MoE2 TFLOPS", "MoE2 TB/s", "MoE1+2 us")
print("\n[DSv4-Pro MoE ubench summary]")
print("| " + " | ".join(cols) + " |")
print("|" + "|".join(["---:"] * len(cols)) + "|")
for (mode, init, tok), r in sorted(data.items()):
    inter, E = SHAPE[mode]
    # Only experts that actually receive routes stream their weights: the
    # EXPERTS_ACT cap when set, else at most one expert per route.
    E = act if act else min(E, tok * TOPK)
    n = inter * 2
    g1, g2 = r.get("gemm1_us"), r.get("gemm2_us")
    f1, f2 = tok * n * MD * TOPK * 2, TOPK * tok * MD * inter * 2
    b1 = tok * MD * AQ + tok * TOPK * n * BO + E * n * MD * WQ
    b2 = tok * TOPK * inter * AQ + tok * MD * BO + E * MD * inter * WQ
    f = lambda v, s="{:.2f}": s.format(v) if v else "-"
    print(f"| {mode} | {init} | {tok} | {E} | {f(r.get('e2e'))} "
          f"| {f(g1)} | {f(f1 / g1 / 1e6, '{:.1f}') if g1 else '-'} "
          f"| {f(b1 / g1 / 1e6, '{:.3f}') if g1 else '-'} "
          f"| {f(g2)} | {f(f2 / g2 / 1e6, '{:.1f}') if g2 else '-'} "
          f"| {f(b2 / g2 / 1e6, '{:.3f}') if g2 else '-'} "
          f"| {f(g1 + g2) if g1 and g2 else '-'} |")
print(f"\nfull log: {log}")
PYEOF

# Optional: same EP shape via the TP harness (no expert_mask) to isolate
# masked-routing overhead from the grouped GEMM.
# python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
#   --scenario bench --data-format a8w4 --act silu \
#   --model-dim 7168 --inter-dim 3072 --experts "$EXPERTS_EP" --topk 6 \
#   --iters "$ITERS" --no-check-aot-cache --tokens $TOKENS_EP
