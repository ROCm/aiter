#!/usr/bin/env bash
# Exit once >= $1 GPUs (rocm-smi indices) have been idle (use < 5%, VRAM < 3%)
# for two checks 60 s apart; prints the idle indices.
need=${1:-4}
prev=""
while true; do
  idle=$(rocm-smi --showuse --showmemuse --json 2>/dev/null | sed -n "/^{/,\$p" | python3 -c '
import json,sys
d=json.load(sys.stdin); out=[]
for k,v in d.items():
    if not k.startswith("card"): continue
    use=float(v.get("GPU use (%)",100)); mem=float(v.get("GPU Memory Allocated (VRAM%)",100))
    if use<5 and mem<3: out.append(k[4:])
print(",".join(sorted(out,key=int)))' || true)
  n=$(echo "$idle" | tr ',' '\n' | grep -c . || true)
  if [ "$n" -ge "$need" ] && [ "$idle" = "$prev" ]; then
    echo "IDLE GPUs (rocm-smi idx): $idle at $(date)"; rocm-smi --showpids 2>/dev/null | grep -E '^[0-9]' || true; exit 0
  fi
  prev=$idle; sleep 60
done
