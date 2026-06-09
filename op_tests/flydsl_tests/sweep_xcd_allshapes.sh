#!/bin/bash
# Per-shape (W,C) device-time sweep for the XCD remap, all 4 headline shapes.
# Control = jagged_dense_bmm_gen (no remap) via bench_headline_worker.py.
# Run inside container with HIP_VISIBLE_DEVICES set by caller.
set -u
cd /home/anguyenh/aiter/op_tests/flydsl_tests
MI=7680
W=8
CGRID="16 32 60 120 240"

run_dt() {  # tag, cmd... -> prints p10 us
  local tag="$1"; shift
  local td="/tmp/rp_${tag}_$$"
  rm -rf "$td"
  rocprofv3 --kernel-trace -d "$td" -- "$@" >/dev/null 2>&1
  python3 read_us2.py "$td" jdbba p10
  rm -rf "$td"
}

for shape in "120 256 256" "120 512 512" "1024 256 256" "1024 512 512"; do
  set -- $shape; B=$1; D=$2; KOUT=$3
  echo "===== B=$B D=$D Kout=$KOUT Mi=$MI ====="
  base=$(run_dt "base_${B}_${D}" python bench_headline_worker.py flydsl $B $D $KOUT $MI)
  echo "  baseline(gen)        p10=${base} us"
  for C in $CGRID; do
    us=$(run_dt "w${W}c${C}_${B}_${D}" python bench_headline_worker_xcd.py $B $D $KOUT $MI $W $C)
    echo "  xcd W=$W C=${C}          p10=${us} us"
  done
done
