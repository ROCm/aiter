#!/bin/bash
# Benchmark: fused MegaMoE TP vs split e2e (AG/RS and AR/AR), all models x tokens.
# Single-process peer-access harness (-s): torchrun cross-process GPU sync is
# pathologically slow on this node. Uses 8 GPUs when all 8 are idle, else 4
# idle GPUs (override with NP=4|8 or GPUS=0,1,2,3). Extra args pass through
# to bench_megamoe_tp.sh (e.g. -m "dsv3 glm5" -t "256 512" -c ar_ar -o outdir).
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-$(command -v python3)}
if [ -z "${NP:-}" ]; then
  NP=$("$PY" - <<'PYEOF'
import json, subprocess, time
busy = set()
for _ in range(5):
    d = json.loads(subprocess.run(["rocm-smi", "--showuse", "--showmemuse", "--json"],
                                  capture_output=True, text=True).stdout)
    for k, v in d.items():
        if k.startswith("card") and (float(v.get("GPU use (%)", 0)) >= 5
                                     or float(v.get("GPU Memory Allocated (VRAM%)", 0)) >= 2):
            busy.add(k)
    time.sleep(0.4)
print(8 if not busy else 4)
PYEOF
)
fi
echo "[run_bench] using $NP GPUs"
exec bash "$DIR/bench_megamoe_tp.sh" -s -n "$NP" "$@"
