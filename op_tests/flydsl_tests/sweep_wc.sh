#!/bin/bash
# Usage: sweep_wc.sh <B> <D> <Kout> <Mi> "<W list>" "<C list>"
# Runs the wc clone bench worker under rocprof for each (W,C), prints p10.
set -u
B=$1; D=$2; Kout=$3; Mi=$4; WLIST=$5; CLIST=$6
TAG="${B}_${D}_${Kout}"
for W in $WLIST; do
  for C in $CLIST; do
    OUT=/tmp/sw_${TAG}_W${W}_C${C}
    rm -rf "$OUT"
    rocprofv3 --kernel-trace -d "$OUT" -- python bench_headline_worker_wc.py $B $D $Kout $Mi $W $C >/dev/null 2>&1
    P10=$(python read_us2.py "$OUT" jdbba p10 2>/dev/null)
    echo "B${B} D${D} Kout${Kout} Mi${Mi} W=${W} C=${C} p10=${P10}"
  done
done
