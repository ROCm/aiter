#!/bin/bash
# bench_m31_sh8.sh <tag> ["<modes>"] ["<tokens>"] [base|mega|both]
# MiniMax-M3.1 TP4 with the A8W8 shared expert: baselines (B1 sep, B2 fuse) and mega (sh8) on 4 GPUs,
# then the speedup table (data_1114/perf_<tag>_m31sh8.csv). One GPU job at a time; a run another job
# overlaps is aborted and repeated (crun.sh); dmesg is checked for new faults after every step.
# Before the first 4-GPU mega run of a changed kernel: 1-GPU validation with tp1g.sh (see README).
TAG=$1; MODES=${2:-"ag_rs ar"}; TOK=${3:-"16 32 64 128 256 512 16384 32768"}; WHAT=${4:-both}
R=/home/binding2/moe_1026/data_1114/raw; OUT=/home/binding2/moe_1026/data_1114/perf_${TAG}_m31sh8.csv
[ $WHAT = both ] && rm -f $OUT
for MO in $MODES; do
  if [ $WHAT != mega ]; then
    echo "#### baseline $MO $(date +%T)"
    bash /tmp/mmtp/bench_sh8_base.sh $MO ${TAG}_base_$MO "$TOK" | grep -v "^T=" || exit 3
  fi
  if [ $WHAT != base ]; then
    echo "#### mega sh8 $MO $(date +%T)"
    MM_EXTRA="--sh8" bash /tmp/mmtp/sweepm.sh m31 $MO ${TAG}_mega_$MO "$TOK" "" | grep -E "STOP|DIRTY|WRONG|rc=[1-9]|BUSY"
  fi
  python3 /tmp/mmtp/tab_sh8.py $MO $R/${TAG}_base_${MO}_sep.log $R/${TAG}_base_${MO}_fuse.log $R/${TAG}_mega_${MO}_1.log $OUT
done
echo "DONE $(date +%T) dmesg faults=$(bash /tmp/mmtp/dmf.sh)  csv: $OUT"
