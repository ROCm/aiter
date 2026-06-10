#!/bin/bash
# Warp-layout sweep on the D512 shapes + D256 re-check, p10 device us.
set -u
cd /home/anguyenh/aiter
export HIP_VISIBLE_DEVICES=4
export FLYDSL_RUNTIME_ENABLE_CACHE=0
MI=7680
OUTBASE=/home/anguyenh/aiter/op_tests/flydsl_tests/_warp_sweep_out
rm -rf "$OUTBASE"; mkdir -p "$OUTBASE"

bench_warps() {  # B D Kout n_warps block_n
  local B=$1 D=$2 K=$3 NW=$4 BN=$5
  local tag="warps_B${B}_D${D}_K${K}_nw${NW}_bn${BN}"
  local out="$OUTBASE/$tag"
  rocprofv3 --kernel-trace -d "$out" -- \
    python op_tests/flydsl_tests/bench_headline_worker_warps.py flydsl $B $D $K $MI $NW $BN >/dev/null 2>&1
  local us=$(python op_tests/flydsl_tests/read_us2.py "$out" jdbba p10)
  echo "B${B}_D${D}_K${K} | nw=${NW} bn=${BN} | p10=${us}"
}

bench_baseline() {  # B D Kout  -- production gen, control
  local B=$1 D=$2 K=$3
  local tag="base_B${B}_D${D}_K${K}"
  local out="$OUTBASE/$tag"
  rocprofv3 --kernel-trace -d "$out" -- \
    python op_tests/flydsl_tests/bench_headline_worker.py flydsl $B $D $K $MI >/dev/null 2>&1
  local us=$(python op_tests/flydsl_tests/read_us2.py "$out" jdbba p10)
  echo "B${B}_D${D}_K${K} | PRODUCTION gen baseline | p10=${us}"
}

echo "===== D512 SHAPES ====="
for shape in "120 512 512" "1024 512 512"; do
  set -- $shape; B=$1 D=$2 K=$3
  bench_baseline $B $D $K
  bench_warps $B $D $K 4 128
  bench_warps $B $D $K 8 128
  bench_warps $B $D $K 8 256
  bench_warps $B $D $K 16 256
  bench_warps $B $D $K 16 512
  echo ""
done

echo "===== D256 SHAPES (re-check) ====="
for shape in "120 256 256" "1024 256 256"; do
  set -- $shape; B=$1 D=$2 K=$3
  bench_baseline $B $D $K
  bench_warps $B $D $K 4 128
  bench_warps $B $D $K 8 128
  bench_warps $B $D $K 8 256
  bench_warps $B $D $K 16 256
  echo ""
done
