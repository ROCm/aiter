#!/bin/bash
# gwait.sh <max minutes> [gpus...]: wait until the GPUs are free (no process, <5% use), polling every 5 s
MAXM=$1; shift; G=${@:-0 1 2 3}
for _i in $(seq $((MAXM * 12))); do
  timeout 90 bash /home/binding2/moe_1026/scripts/gpu_free.sh $G >/dev/null && exit 0
  sleep 5
done
echo "GPU BUSY after ${MAXM} min"; exit 2
