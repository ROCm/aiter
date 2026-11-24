#!/bin/bash
# full same-batch table after the upstream merge: 3 models x 2 modes x target tokens (one GPU job at a time)
TAG=${1:-r200}; WHAT=${2:-both}
T="8 16 32 64 128 256 16384 32768"
OUT=/home/binding2/moe_1026/data_1114/perf_${TAG}_all.csv
rm -f $OUT
for MO in ag_rs ar; do
  for M in dsv41 m31 kimi31; do
    echo "#### $M $MO $(date +%T)"
    bash /tmp/mmtp/bench_1120.sh $M $MO ${TAG}_${M}_$MO "$T" $WHAT 2>&1 | grep -v "^tokens\|^ *[0-9]* |" | tail -6
    python3 /tmp/mmtp/tab_1120.py $M $MO /home/binding2/moe_1026/data_1114/raw/${TAG}_${M}_${MO}_base.log /home/binding2/moe_1026/data_1114/raw/${TAG}_${M}_${MO}_mega.log $OUT
    sleep 5
  done
done
echo "ALL DONE $(date +%T)"
