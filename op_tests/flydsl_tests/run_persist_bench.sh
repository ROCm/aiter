#!/bin/bash
set -e
cd /home/anguyenh/aiter
export HIP_VISIBLE_DEVICES=5
export FLYDSL_RUNTIME_ENABLE_CACHE=0
T=op_tests/flydsl_tests
RD=$T/read_us2.py

prof () { # outdir  script  args...
  local out=$1; shift
  local script=$1; shift
  rm -rf "$out"
  rocprofv3 --kernel-trace -d "$out" -- python "$script" "$@" >/dev/null 2>&1
}

echo "###### UNIFORM Mi=7680 (regression: persist kernel-only vs baseline) ######"
for shape in "120 256 256" "120 512 512" "1024 256 256" "1024 512 512"; do
  set -- $shape; B=$1; D=$2; Kout=$3
  prof $T/_u_base_${B}_${D} $T/bench_headline_worker_gen.py $B $D $Kout 7680 8 60 2
  PB=$(python $RD $T/_u_base_${B}_${D} jdbba p10)
  prof $T/_u_pers_${B}_${D} $T/bench_headline_worker_persist.py $B $D $Kout 7680
  PP=$(python $RD $T/_u_pers_${B}_${D} jdbba p10)
  echo "UNIFORM B${B}_D${D}: baseline=${PB} us   persist(kernel-only)=${PP} us"
done

echo ""
echo "###### SKEWED (power-law ~27% empty): baseline vs persist ######"
for shape in "120 256 256" "120 512 512" "1024 256 256" "1024 512 512"; do
  set -- $shape; B=$1; D=$2; Kout=$3
  prof $T/_s_base_${B}_${D} $T/bench_skew_worker_persist.py $B $D $Kout 7680 baseline_gen
  SB=$(python $RD $T/_s_base_${B}_${D} jdbba p10)
  prof $T/_s_pers_${B}_${D} $T/bench_skew_worker_persist.py $B $D $Kout 7680 persist_kernel
  SP=$(python $RD $T/_s_pers_${B}_${D} jdbba p10)
  echo "SKEWED B${B}_D${D}: baseline=${SB} us   persist(kernel-only)=${SP} us"
done

echo ""
echo "###### SKEWED wall-clock (persist_kernel vs persist_host vs baseline) ######"
for shape in "120 256 256" "1024 512 512"; do
  set -- $shape; B=$1; D=$2; Kout=$3
  python $T/bench_skew_worker_persist.py $B $D $Kout 7680 baseline_gen   2>/dev/null | grep WALL
  python $T/bench_skew_worker_persist.py $B $D $Kout 7680 persist_kernel 2>/dev/null | grep WALL
  python $T/bench_skew_worker_persist.py $B $D $Kout 7680 persist_host   2>/dev/null | grep WALL
done
