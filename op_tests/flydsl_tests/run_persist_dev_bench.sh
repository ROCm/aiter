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

echo "###### KERNEL-ONLY device time (p10 us), SKEW: baseline vs persist(old) vs persist_dev ######"
for shape in "120 256 256" "120 512 512" "1024 256 256" "1024 512 512"; do
  set -- $shape; B=$1; D=$2; Kout=$3
  prof $T/_sd_base_${B}_${D} $T/bench_skew_worker_persist_dev.py $B $D $Kout 7680 baseline_gen 1234 skew
  SB=$(python $RD $T/_sd_base_${B}_${D} jdbba p10)
  prof $T/_sd_pold_${B}_${D} $T/bench_skew_worker_persist_dev.py $B $D $Kout 7680 persist_host 1234 skew
  SO=$(python $RD $T/_sd_pold_${B}_${D} jdbba p10)
  prof $T/_sd_pdev_${B}_${D} $T/bench_skew_worker_persist_dev.py $B $D $Kout 7680 persist_dev_kernel 1234 skew
  SD=$(python $RD $T/_sd_pdev_${B}_${D} jdbba p10)
  echo "SKEW  B${B}_D${D}: baseline=${SB}  persist_old=${SO}  persist_dev=${SD}  us"
done

echo ""
echo "###### KERNEL-ONLY device time (p10 us), UNIFORM: baseline vs persist_dev ######"
for shape in "120 256 256" "120 512 512" "1024 256 256" "1024 512 512"; do
  set -- $shape; B=$1; D=$2; Kout=$3
  prof $T/_ud_base_${B}_${D} $T/bench_headline_worker_gen.py $B $D $Kout 7680 8 60 2
  UB=$(python $RD $T/_ud_base_${B}_${D} jdbba p10)
  prof $T/_ud_pdev_${B}_${D} $T/bench_headline_worker_persist_dev.py $B $D $Kout 7680
  UD=$(python $RD $T/_ud_pdev_${B}_${D} jdbba p10)
  echo "UNIF  B${B}_D${D}: baseline=${UB}  persist_dev=${UD}  us"
done
