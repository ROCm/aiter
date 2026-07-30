#!/usr/bin/env bash
# GEMM1/GEMM2 TDM tile+buffer sweep. Run INSIDE container hyg_fyd2.
#   cd /data/yanguahe/code/wk_sp1/aiter && bash my_code/sweep_tdm.sh [tag ...]
# With no args runs the full list; with args runs only the named configs.
# Results -> my_code/sweep_tdm/summary.tsv
set -u

cd "$(dirname "$0")/.." || exit 1
OUT=my_code/sweep_tdm
mkdir -p "$OUT"
SUM="$OUT/summary.tsv"

export ENABLE_CK=0
export AITER_MOE_EXPERT_BALANCE=true
export AITER_LOG_MORE=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
export AITER_FORCE_GFX1250=1
unset FLYDSL_DUMP_IR FLYDSL_DUMP_DIR

ONLY=" $* "
[ -f "$SUM" ] || printf 'tag\tg1cfg\tg2cfg\tstatus\tgemm1_us\tgemm2_us\te2e_us\trel_l2\tlds1\tlds2\n' > "$SUM"

run() {  # tag tm tn tk nb tm2 tn2 tk2 nb2
  local tag=$1 tm=$2 tn=$3 tk=$4 nb=$5 tm2=$6 tn2=$7 tk2=$8 nb2=$9
  if [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $tag "* ]]; then return; fi
  local log="$OUT/$tag.log"
  echo "=== [$tag] g1=${tm}x${tn}x${tk}_b${nb}  g2=${tm2}x${tn2}x${tk2}_b${nb2}"
  rm -rf ~/.flydsl/cache/* 2>/dev/null

  AITER_TDM_TILE_M=$tm AITER_TDM_TILE_N=$tn AITER_TDM_TILE_K=$tk \
  AITER_TDM_NUM_BUFFERS=$nb \
  AITER_TDM_TILE_M2=$tm2 AITER_TDM_TILE_N2=$tn2 AITER_TDM_TILE_K2=$tk2 \
  AITER_TDM_NUM_BUFFERS2=$nb2 \
  timeout 2400 python op_tests/test_flydsl_grouped_gemm_gfx1250.py \
    --scenario bench --data-format a8w4 --layout gugu \
    --experts 384 --tokens 4096 --topk 6 --iters 100 \
    --model-dim 7168 --inter-dim 768 --act silu --real-gemm --no-check-aot \
    > "$log" 2>&1
  local rc=$?

  # per-kernel device_time_avg straight from the torch-profiler table (last block)
  local g1 g2 e2e l2 st
  g1=$(grep -E 'gemm_a8w4_tdm_.*_silu_'   "$log" | tail -1 | awk '{print $(NF-2)}' | tr -d ,)
  g2=$(grep -E 'gemm_a8w4_tdm_.*_noact_'  "$log" | tail -1 | awk '{print $(NF-2)}' | tr -d ,)
  e2e=$(grep -oE 'end-to-end us = [0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')
  l2=$(grep -oE 'rel_l2=[0-9.e+-]+' "$log" | tail -1 | cut -d= -f2)
  local a; a=$(grep -oE 'ARENA=[0-9]+' "$log")
  local l1b l2b
  l1b=$(echo "$a" | sed -n 1p | cut -d= -f2); l2b=$(echo "$a" | sed -n 2p | cut -d= -f2)

  st=OK
  [ $rc -ne 0 ] && st="FAIL(rc=$rc)"
  [ -z "$g1" ] && { st="$st/no-g1"; g1=-1; }
  [ -z "$g2" ] && { st="$st/no-g2"; g2=-1; }
  grep -qiE 'Traceback|out of memory|LDS.*exceed|insufficient' "$log" && st="$st/err"

  printf '%s\t%dx%dx%d_b%d\t%dx%dx%d_b%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$tm" "$tn" "$tk" "$nb" "$tm2" "$tn2" "$tk2" "$nb2" \
    "$st" "$g1" "$g2" "${e2e:--}" "${l2:--}" "${l1b:--}" "${l2b:--}" >> "$SUM"
  echo "    -> $st gemm1=${g1}us gemm2=${g2}us e2e=${e2e:--} rel_l2=${l2:--} lds=${l1b:--}/${l2b:--}"
}

#    tag              g1: tm  tn   tk  nb    g2: tm2 tn2  tk2 nb2
run base              16 256 256 2      16 512 128 2
# A: deepen GEMM1 pipeline (LDS 77K -> 115K / 154K, occupancy 2 -> 1)
run g1_nb3            16 256 256 3      16 512 128 2
run g1_nb4            16 256 256 4      16 512 128 2
# B: halve tile_n to buy buffer depth while keeping >=2 blocks/CU
run g1_n128_nb3       16 128 256 3      16 512 128 2
run g1_n128_nb4       16 128 256 4      16 512 128 2
# NOT run: 16 128 256 6 -- wedges the GPU (MES stops answering REMOVE_QUEUE, ASIC
# reset then fails, machine needs a reboot). Reproduced twice. Root cause unknown:
# compile is fine (7.7s) and a standalone kernel issuing 20 TDM ops with
# s_wait_tensorcnt 0x10 runs clean, so it is not TDM over-subscription alone.
# Not worth chasing -- this tile_n=128 line tops out at 496us vs 378us for
# g1_m64_nb3. Do not re-add without a containment plan.
# C: bigger tile_k lengthens the per-tile compute window
run g1_k512_nb2       16 256 512 2      16 512 128 2
run g1_n128_k512_b3   16 128 512 3      16 512 128 2
run g1_n128_k512_b4   16 128 512 4      16 512 128 2
# D: GEMM2 side (same 58% WAIT_tensor bottleneck)
run g2_nb3            16 256 256 2      16 512 128 3
run g2_nb4            16 256 256 2      16 512 128 4
run g2_n256_nb4       16 256 256 2      16 256 128 4
run g2_n256_k256_b3   16 256 256 2      16 256 256 3
# E: combine the winners
run both_nb3          16 256 256 3      16 512 128 3
# F: tile_m=64 == rows/expert, so the weight slab is read once instead of 4x.
#    Per-unit-M-work TDM traffic drops ~3x (39040 -> 12928 B per K-tile).
run g1_m64_nb2        64 256 256 2      16 512 128 2
run g1_m64_nb3        64 256 256 3      16 512 128 2
run g1_m64_n128_nb3   64 128 256 3      16 512 128 2
run g1_m64_n128_nb4   64 128 256 4      16 512 128 2
run g1_m32_nb3        32 256 256 3      16 512 128 2
run g1_m128_nb2      128 256 256 2      16 512 128 2
# G: combine the per-GEMM winners (m64_nb3 for gemm1, nb3 for gemm2 -- never yet
#    measured together), and probe tile_k=512 on top of tile_m=64.
run best_m64_g2nb3    64 256 256 3      16 512 128 3
run best_m64b2_g2nb3  64 256 256 2      16 512 128 3
run g1_m64_k512_nb2   64 256 512 2      16 512 128 3
run g1_m64_nb4        64 256 256 4      16 512 128 3
# gemm2 tile_m must stay 16: tile_m2=64 would raise align_m and add empty tiles,
# but check whether matching it to gemm1 helps now that gemm1 is 3x faster.
run g2_m64_nb3        64 256 256 3      64 512 128 3

echo
echo "================ SUMMARY ================"
column -t -s $'\t' "$SUM" 2>/dev/null || cat "$SUM"
