#!/bin/bash
# Fill missing passes from decode_profile_run.sh, retrying idle check up to 6x (10 s apart).
export ENABLE_CK=0 HIP_VISIBLE_DEVICES=2 FLYDSL_RUNTIME_ENABLE_CACHE=0
export PYTHONPATH=/home/jograner/projects/aiter/unified-attention-gemma4:/home/jograner/projects/aiter/unified-attention-gemma4/profiling-data
cd /home/jograner/projects/aiter/unified-attention-gemma4
R=/tmp/ua_repro/decode_prof
GROUPS_LIST=(
"SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES GRBM_GUI_ACTIVE"
"SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_SALU SQ_INSTS_VMEM_RD SQ_INSTS_LDS SQ_INSTS_SMEM"
"SQ_WAIT_INST_ANY SQ_WAIT_ANY SQ_INST_CYCLES_VMEM_RD SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_ANY SQ_VALU_MFMA_BUSY_CYCLES"
"TCC_HIT_sum TCC_MISS_sum TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCP_TCC_READ_REQ_sum"
"FETCH_SIZE"
"SQ_LDS_BANK_CONFLICT SQ_LEVEL_WAVES SQ_ACCUM_PREV_HIRES"
"OccupancyPercent MeanOccupancyPerCU MeanOccupancyPerActiveCU"
)
idle() { for t in 1 2 3 4 5 6; do gpu-idle-check 2 >/dev/null 2>&1 && return 0; sleep 10; done; return 1; }
for case in "64 32" "64 64" "16 64" "16 32"; do
 set -- $case; b=$1; p=$2
 for be in flydsl triton; do
  tag=${be}_B${b}_p${p}
  if ! ls $R/$tag/trace/*/kt_kernel_trace.csv >/dev/null 2>&1 && ! ls $R/$tag/trace/kt_kernel_trace.csv >/dev/null 2>&1; then
   idle && timeout 250 rocprofv3 --kernel-trace --output-format csv -d $R/$tag/trace -o kt -- python3 profiling-data/decode_profile.py --backend $be --page $p --batch $b > $R/$tag.trace.log 2>&1 || echo "SKIP $tag trace"
  fi
  i=0
  for g in "${GROUPS_LIST[@]}"; do
   i=$((i+1))
   ls $R/$tag/g$i/*/pmc_counter_collection.csv >/dev/null 2>&1 && continue
   ls $R/$tag/g$i/pmc_counter_collection.csv >/dev/null 2>&1 && continue
   idle && timeout 250 rocprofv3 --pmc $g --output-format csv -d $R/$tag/g$i -o pmc -- python3 profiling-data/decode_profile.py --backend $be --page $p --batch $b > $R/$tag.g$i.log 2>&1 || echo "SKIP $tag g$i"
  done
 done
done
echo FILLDONE
