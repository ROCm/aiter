#!/bin/bash
# Interleaved same-session control: baseline (gen) vs DTLA, GPU 6, Mi=7680, p10.
set -e
cd /home/anguyenh/aiter
export HIP_VISIBLE_DEVICES=6
export FLYDSL_RUNTIME_ENABLE_CACHE=0
declare -A CMAP=( ["256"]=32 ["512"]=60 )
for shape in "120 256 256" "120 512 512" "1024 256 256" "1024 512 512"; do
  set -- $shape
  B=$1; D=$2; Kout=$3
  C=${CMAP[$D]}

  OUTB=/home/anguyenh/aiter/op_tests/flydsl_tests/_base_${B}_${D}
  rm -rf $OUTB
  rocprofv3 --kernel-trace -d $OUTB -- python op_tests/flydsl_tests/bench_headline_worker_gen.py $B $D $Kout 7680 8 $C 2 >/dev/null 2>&1
  PB=$(python op_tests/flydsl_tests/read_us2.py $OUTB jdbba p10)

  OUTD=/home/anguyenh/aiter/op_tests/flydsl_tests/_dtla_${B}_${D}
  rm -rf $OUTD
  rocprofv3 --kernel-trace -d $OUTD -- python op_tests/flydsl_tests/bench_headline_worker_dtla.py $B $D $Kout 7680 8 $C >/dev/null 2>&1
  PD=$(python op_tests/flydsl_tests/read_us2.py $OUTD jdbba p10)

  SPD=$(python -c "b=$PB; d=$PD; print(f'{b/d:.3f}x') if d>0 else print('nan')")
  echo "B${B}_D${D}: baseline=${PB} us  dtla=${PD} us  speedup=${SPD}  (W=8 C=$C)"
done
