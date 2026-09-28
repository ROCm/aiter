#!/bin/bash
# Unit tests: FlyDSL mega_moe_tp accuracy / function.
#   glm5  : W8A8 and A4W4 kernels at 1, 2, 4 tokens against the torch golden
#           (norm / logits / top-k / probs / mids / all-reduced out)
#   kimi3 : A4W4 Kimi-K3 kernel at 1, 2, 4 tokens against its torch golden
#   e2e   : fused (multi-launch batches) vs split MoE outputs, same weights and
#           tokens, batch sizes 1 .. 2048 (no timing), plus the poll watchdogs
# One process drives all TP GPUs (peer access). Checks GPU state first: 8 GPUs
# when all 8 are idle, else 4 idle GPUs (NP=4|8, GPUS=... override).
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH="1 2 3 4 8 16 32 64 128 256 512 1024 2048"
OUT="$DIR/mega_moe_tp_results/ut_$(date +%m%d_%H%M%S)"
while getopts "b:o:h" opt; do
  case $opt in
    b) BATCH=$OPTARG ;;
    o) OUT=$OPTARG ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
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
echo "[run_mega_moe_tp_ut] GPUs $GPUS (TP$NP), results: $OUT"
cd "$DIR/op_tests/multigpu_tests"
fail=0
"$PY" test_flydsl_mega_moe_tp.py --model glm5 -q 8 4 -s 1 2 4 --proto 0 1 --tp "$NP" --no-perf \
  > "$OUT/glm5_tp$NP.log" 2>&1 || fail=1
echo "glm5 kernels (W8A8, A4W4): $([ $fail = 0 ] && echo PASS || echo FAIL)"
grep -E "^\|" "$OUT/glm5_tp$NP.log" | head -20
"$PY" test_flydsl_mega_moe_tp.py --model kimi3 glm5 -s 1 2 4 -b $BATCH --tp "$NP" --no-perf \
  --csv "$OUT/e2e_tp$NP.csv" > "$OUT/kimi3_e2e_tp$NP.log" 2>&1 || fail=2
echo "kimi3 kernel + fused vs split e2e: $([ $fail != 2 ] && echo PASS || echo FAIL)"
grep -E "^\|" "$OUT/kimi3_e2e_tp$NP.log"
echo "logs: $OUT"
[ $fail = 0 ] && echo "ALL PASS" || echo "SOME TESTS FAILED"
exit $fail
