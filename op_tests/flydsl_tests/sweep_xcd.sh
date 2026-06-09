#!/bin/bash
# Sweep XCD (W,C) configs: device time (kernel-trace) + cache PMC.
# Usage: sweep_xcd.sh   (run inside container, HIP_VISIBLE_DEVICES=4 set by caller)
set -u
cd /home/anguyenh/aiter/op_tests/flydsl_tests
B=1024; D=512; KOUT=512; MI=7680
PMC="TCC_HIT TCC_MISS TCC_EA0_RDREQ TCC_EA0_RDREQ_DRAM TCC_EA0_RDREQ_DRAM_32B"

run_one() {
  local tag="$1"; shift
  local cmd=("$@")
  local td="/tmp/rp_${tag}"
  rm -rf "$td"
  rocprofv3 --kernel-trace -d "$td" -- "${cmd[@]}" >/dev/null 2>&1
  local us=$(python3 read_us2.py "$td" jdbba p10)
  local tp="/tmp/rppmc_${tag}"
  rm -rf "$tp"
  rocprofv3 --pmc $PMC -d "$tp" -- "${cmd[@]}" >/dev/null 2>&1
  python3 parse_pmc.py "$tp" "$us" "$tag"
}

# baseline control (jagged_dense_bmm_gen)
run_one "base" python bench_headline_worker.py flydsl $B $D $KOUT $MI

# xcd sweep
for wc in "4 16" "8 16" "8 32" "5 25" "8 64" "8 120" "8 240" "4 240" "8 480"; do
  set -- $wc; W=$1; C=$2
  run_one "w${W}c${C}" python bench_headline_worker_xcd.py $B $D $KOUT $MI $W $C
done
