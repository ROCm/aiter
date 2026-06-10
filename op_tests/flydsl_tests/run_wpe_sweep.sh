#!/bin/bash
set -e
cd /home/anguyenh/aiter
export HIP_VISIBLE_DEVICES=6
export FLYDSL_RUNTIME_ENABLE_CACHE=0
for shape in "120 256 256" "120 512 512" "1024 256 256" "1024 512 512"; do
  set -- $shape
  B=$1; D=$2; Kout=$3
  line="B${B}_D${D}:"
  for WPE in 0 1 2 3 4; do
    OUT=/home/anguyenh/aiter/op_tests/flydsl_tests/_wpe_sweep_${B}_${D}_w${WPE}
    rm -rf $OUT
    rocprofv3 --kernel-trace -d $OUT -- python op_tests/flydsl_tests/bench_headline_worker_wpe.py $B $D $Kout 7680 $WPE >/dev/null 2>&1
    P=$(python op_tests/flydsl_tests/read_us2.py $OUT jdbba p10)
    line="$line  w${WPE}=${P}"
  done
  echo "$line"
done
