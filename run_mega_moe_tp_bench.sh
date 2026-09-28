#!/bin/bash
# Benchmark: FlyDSL mega_moe_tp vs the split MoE e2e, GLM-5 (W8A8) and
# Kimi-K3 (A4W4), over batch sizes 1 .. 2048 tokens.
#   fused = ceil(T / 4) decode launches (4, 2, 1 tokens each) of the fused
#           router + top-k + experts + TP all-reduce kernel
#   split = [RMSNorm] + router GEMM + aiter biased top-k + aiter fused_moe
#           (tuned) + all-reduce (+ residual)
# Per-forward time over CUDA graphs (every rank replayed from its own thread),
# slowest rank, best of rounds. One process drives all TP GPUs (peer access).
# Checks GPU state first: 8 GPUs when all 8 are idle, else 4 idle GPUs.
# Options: -m "glm5 kimi3"  -b "1 2 ... 2048"  -o out_dir   (NP=4|8, GPUS=... override)
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="glm5 kimi3"
BATCH="1 2 3 4 8 16 32 64 128 256 512 1024 2048"
OUT="$DIR/mega_moe_tp_results/bench_$(date +%m%d_%H%M%S)"
while getopts "m:b:o:h" opt; do
  case $opt in
    m) MODELS=$OPTARG ;;
    b) BATCH=$OPTARG ;;
    o) OUT=$OPTARG ;;
    *) sed -n '2,13p' "$0"; exit 2 ;;
  esac
done
PY=${PYTHON:-$(command -v python3)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
pick_gpus() {
  "$PY" - "$1" <<'PYEOF'
import json, subprocess, sys, time
want, busy, d = sys.argv[1], set(), {}
for _ in range(5):
    d = json.loads(subprocess.run(["rocm-smi", "--showuse", "--showmemuse", "--json"],
                                  capture_output=True, text=True).stdout)
    for k, v in d.items():
        if k.startswith("card") and (float(v.get("GPU use (%)", 0)) >= 5
                                     or float(v.get("GPU Memory Allocated (VRAM%)", 0)) >= 2):
            busy.add(int(k[4:]))
    time.sleep(0.4)
free = sorted(int(k[4:]) for k in d if k.startswith("card") and int(k[4:]) not in busy)
n = int(want) if want else (8 if len(free) >= 8 else 4)
if len(free) < n:
    sys.exit(1)
print(",".join(map(str, free[:n])))
PYEOF
}
if [ -z "${GPUS:-}" ]; then
  GPUS=$(pick_gpus "${NP:-}") || { echo "not enough idle GPUs"; exit 1; }
fi
export HIP_VISIBLE_DEVICES=$GPUS
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
echo "[run_mega_moe_tp_bench] GPUs $GPUS (TP$NP), models: $MODELS, batches: $BATCH, results: $OUT"
cd "$DIR/op_tests/multigpu_tests"
"$PY" test_flydsl_mega_moe_tp.py --model $MODELS -s 1 -b $BATCH --tp "$NP" \
  --csv "$OUT/fused_vs_split_tp$NP.csv" 2>&1 | tee "$OUT/bench_tp$NP.log" \
  | grep -E "^\||markdown|Error|failed"
exit "${PIPESTATUS[0]}"
