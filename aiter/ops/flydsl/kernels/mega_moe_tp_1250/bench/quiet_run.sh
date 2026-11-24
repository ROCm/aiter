#!/bin/bash
# quiet_run.sh <minutes quiet> <max hours> <cmd...>: start cmd once no GPU has had any process for
# <minutes quiet> minutes in a row (another tenant runs bursts of 4-GPU jobs; a persistent kernel next to
# them only times out). Gives up after <max hours>.
Q=$1; H=$2; shift 2
need=$((Q * 3)); ok=0
for _i in $(seq $((H * 180))); do
  if timeout 90 bash /home/binding2/moe_1026/scripts/gpu_free.sh 0 1 2 3 >/dev/null; then ok=$((ok + 1)); else ok=0; fi
  if [ $ok -ge $need ]; then echo "quiet for ${Q} min at $(date +%T): starting"; exec "$@"; fi
  sleep 20
done
echo "never quiet for ${Q} min in ${H} h"; exit 2
