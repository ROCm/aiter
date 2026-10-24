#!/usr/bin/env bash
# M2 / M3 run recipes for Evidence (single node, GPUs 0..3 free). usage: BIN=/build/dir bash run_bench.sh [m2|m3|all] > out.tsv
set -euo pipefail
BIN="${BIN:?set BIN to the build directory}"
what="${1:-all}"
if [ "${what}" = "m2" ] || [ "${what}" = "all" ]; then
  echo "## M2-a bytes x WGs (512 thr, 1 release, 3 destinations, SC0|SC1 stores)"
  "${BIN}/m2_cu_push" --bytes 8192,16384,32768,65536 --wgs 24,48 --threads 512 --rel 1 --ndst 3 --mode 0
  echo "## M2-b threads per WG at equal total bytes (24x512 vs 48x256)"
  "${BIN}/m2_cu_push" --bytes 32768 --wgs 24 --threads 512 --rel 1 --ndst 3 --mode 0
  "${BIN}/m2_cu_push" --bytes 16384 --wgs 48 --threads 256 --rel 1 --ndst 3 --mode 0
  echo "## M2-c releases per WG (r18: 1; straddlers 2; 6 releases/XCC for 48 WGs)"
  "${BIN}/m2_cu_push" --bytes 32768 --wgs 24 --threads 512 --rel 0,1,2,3 --ndst 3 --mode 0
  "${BIN}/m2_cu_push" --bytes 16384 --wgs 48 --threads 256 --rel 0,1,2 --ndst 3 --mode 0
  echo "## M2-d link-bound vs concurrency-bound: fixed total bytes, WG count sweep (1 destination, then 3)"
  "${BIN}/m2_cu_push" --bytes 786432,393216,196608,98304,49152 --wgs 1,2,4,8,16 --threads 512 --rel 1 --ndst 1 --mode 0
  "${BIN}/m2_cu_push" --bytes 32768 --wgs 6,12,24,48,96 --threads 512 --rel 1 --ndst 3 --mode 0
  echo "## M2-e store flavour (SC0|SC1 vs plain vs non-temporal), 24x512, 32 KiB"
  "${BIN}/m2_cu_push" --bytes 32768 --wgs 24 --threads 512 --rel 1 --ndst 3 --mode 0,1,2
fi
if [ "${what}" = "m3" ] || [ "${what}" = "all" ]; then
  for N in 9216 9856; do for M in 16 32 33 48 63 64; do
    echo "## M3 N=${N} M=${M} SK=3 warm L2"; "${BIN}/m3_gemm_bm" --m "${M}" --n "${N}" --sk 3
  done; done
  echo "## M3 cold-L2 (512 MB flush between launches), N=9216 M=64"; "${BIN}/m3_gemm_bm" --m 64 --n 9216 --sk 3 --cold 1
  echo "## M3 M=128 (BM128 only)"; "${BIN}/m3_gemm_bm" --m 128 --n 9216 --sk 3 --only bm128_pfk3
  echo "## M3 SK=2 control"; "${BIN}/m3_gemm_bm" --m 64 --n 9216 --sk 2
fi