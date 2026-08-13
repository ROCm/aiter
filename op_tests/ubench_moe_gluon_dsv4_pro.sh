#!/bin/bash
# DSv4-Pro gluon (Triton gfx1250) a8w4 MoE ubench: TP + EP shapes.
# Drives op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_cudagraph.py, which
# times moe1 / moe2 / total with do_bench_cudagraph.
#   TP -> 384 experts local, inter 3072/TP
#   EP -> 384/EP local experts, inter 3072
# --shape takes dim2 = 2*inter_dim (the bench derives inter = dim2/2).
#
# Differences vs the FlyDSL ubench (ubench_moe_dsv4_pro.sh): timing is
# cudagraph replay rather than profiler device-time, and buffers are not
# rotated (a graph replays over the same tensors).
#
# Usage: [TP=4] [EP=4] [TOKENS_TP=64] [TOKENS_EP=2048] [EXPERTS_ACT=0] [REP=20]
#        [CONST_VAL=0.5] [LOG=./gluon_ubench.log] op_tests/ubench_moe_gluon_dsv4_pro.sh

set -uo pipefail   # no -e: one failed case must not kill later runs

AITER_DIR="${AITER_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"   # repo root = op_tests/..
TP="${TP:-4}"
EP="${EP:-4}"
TOKENS_TP="${TOKENS_TP:-64}"
TOKENS_EP="${TOKENS_EP:-2048}"
REP="${REP:-20}"       # do_bench_cudagraph target per batch size, in ms
CONST_VAL="${CONST_VAL:-0.5}"   # --const-init value for the const runs
# Experts actually receiving routes; 0 = whatever random routing lands on.
EXPERTS_ACT="${EXPERTS_ACT:-0}"
ACT_ARGS=()
[ "$EXPERTS_ACT" != "0" ] && ACT_ARGS=(--routed-experts "$EXPERTS_ACT")

DIM2_TP=$((2 * 3072 / TP))
EXPERTS_EP=$((384 / EP))
LOG="${LOG:-$PWD/gluon_ubench.log}"
: > "$LOG"

cd "$AITER_DIR" || exit 1

export HIP_VISIBLE_DEVICES=0

echo "== DSv4-Pro gluon MoE ubench | TP=$TP EP=$EP rep=${REP}ms | tokens TP=$TOKENS_TP EP=$TOKENS_EP | experts_act=$EXPERTS_ACT | const=$CONST_VAL"

echo
echo "##### [1/4] mode=TP init=rand 7168/$((DIM2_TP / 2))/384/topk6 a8w4 #####" | tee -a "$LOG"
python3 -u op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_cudagraph.py \
  --shape 7168 "$DIM2_TP" --experts 384 6 --act-dtype fp8 --preshuffle \
  --rep "$REP" "${ACT_ARGS[@]}" --M $TOKENS_TP | tee -a "$LOG"

echo
echo "##### [2/4] mode=TP init=const 7168/$((DIM2_TP / 2))/384/topk6 a8w4 #####" | tee -a "$LOG"
python3 -u op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_cudagraph.py \
  --shape 7168 "$DIM2_TP" --experts 384 6 --act-dtype fp8 --preshuffle \
  --rep "$REP" "${ACT_ARGS[@]}" --const-init "$CONST_VAL" --M $TOKENS_TP | tee -a "$LOG"

echo
echo "##### [3/4] mode=EP init=rand 7168/3072/$EXPERTS_EP/topk6 a8w4 #####" | tee -a "$LOG"
python3 -u op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_cudagraph.py \
  --shape 7168 6144 --experts "$EXPERTS_EP" 6 --act-dtype fp8 --preshuffle \
  --rep "$REP" "${ACT_ARGS[@]}" --M $TOKENS_EP | tee -a "$LOG"

echo
echo "##### [4/4] mode=EP init=const 7168/3072/$EXPERTS_EP/topk6 a8w4 #####" | tee -a "$LOG"
python3 -u op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_cudagraph.py \
  --shape 7168 6144 --experts "$EXPERTS_EP" 6 --act-dtype fp8 --preshuffle \
  --rep "$REP" "${ACT_ARGS[@]}" --const-init "$CONST_VAL" --M $TOKENS_EP | tee -a "$LOG"

echo
echo "== done"

# The bench already reports TFLOPS/TB-s per layer; this just collects both runs
# into one table keyed by mode.
python3 - "$LOG" <<'PYEOF'
import re
import sys

mode, init, rows = None, None, []
for ln in open(sys.argv[1], errors="ignore"):
    if ln.startswith("##### ["):
        kv = dict(p.split("=", 1) for p in ln.split() if "=" in p)
        mode, init = kv.get("mode"), kv.get("init")
    elif ln.startswith("batch:"):
        f = {k: v for k, v in re.findall(r"(\w+)=([\d.]+)", ln)}
        m = re.findall(r"(moe1|moe2|total) ([\d.]+)us ([\d.]+) TF/s ([\d.]+) TB/s", ln)
        b = re.search(r"batch:\s*(\d+)", ln).group(1)
        rows.append((mode, init, b, dict((k, v[1:]) for k, v in [(x[0], x) for x in m]),
                     f.get("routed_experts", "-"), f.get("block_m", "-")))

cols = ("mode", "init", "tokens", "E_act", "block_m", "MoE1 us", "MoE1 TFLOPS", "MoE1 TB/s",
        "MoE2 us", "MoE2 TFLOPS", "MoE2 TB/s", "total us", "total TFLOPS")
print("\n[DSv4-Pro gluon MoE ubench summary]")
print("| " + " | ".join(cols) + " |")
print("|" + "|".join(["---:"] * len(cols)) + "|")
for mode, init, b, m, e, bm in rows:
    g1, g2, tt = m.get("moe1", ["-"] * 3), m.get("moe2", ["-"] * 3), m.get("total", ["-"] * 3)
    print(f"| {mode} | {init} | {b} | {e} | {bm} | {g1[0]} | {g1[1]} | {g1[2]} "
          f"| {g2[0]} | {g2[1]} | {g2[2]} | {tt[0]} | {tt[1]} |")
print(f"\nfull log: {sys.argv[1]}")
PYEOF
