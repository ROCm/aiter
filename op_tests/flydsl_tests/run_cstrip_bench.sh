#!/bin/bash
set -e
cd /home/anguyenh/aiter
S=$1
# shape: B D Kout ; knobs W C
declare -A CMAP=( ["256"]=32 ["512"]=60 )
for shape in "120 256 256" "120 512 512" "1024 256 256" "1024 512 512"; do
  set -- $shape
  B=$1; D=$2; Kout=$3
  C=${CMAP[$D]}
  OUT=/home/anguyenh/aiter/op_tests/flydsl_tests/_cstrip_${B}_${D}_S${S}
  rm -rf $OUT
  rocprofv3 --kernel-trace -d $OUT -- python op_tests/flydsl_tests/bench_headline_worker_cstrip.py $B $D $Kout 7680 8 $C $S >/dev/null 2>&1
  P=$(python op_tests/flydsl_tests/read_us2.py $OUT jdbba p10)
  echo "S=$S B${B}_D${D}: p10=${P} us (W=8 C=$C)"
done
