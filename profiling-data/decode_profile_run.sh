#!/bin/bash
# Sequential rocprofv3 sweep: kernel trace + counter passes per case/backend. Device 2 only.
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
for case in "64 32" "64 64" "16 64" "16 32"; do
 set -- $case; b=$1; p=$2
 for be in flydsl triton; do
  tag=${be}_B${b}_p${p}
  gpu-idle-check 2 >/dev/null || { echo "BUSY $tag trace"; continue; }
  timeout 250 rocprofv3 --kernel-trace --output-format csv -d $R/$tag/trace -o kt -- python3 profiling-data/decode_profile.py --backend $be --page $p --batch $b > $R/$tag.trace.log 2>&1
  i=0
  for g in "${GROUPS_LIST[@]}"; do
   i=$((i+1))
   gpu-idle-check 2 >/dev/null || { echo "BUSY $tag g$i"; continue; }
   timeout 250 rocprofv3 --pmc $g --output-format csv -d $R/$tag/g$i -o pmc -- python3 profiling-data/decode_profile.py --backend $be --page $p --batch $b > $R/$tag.g$i.log 2>&1 || echo "FAIL $tag g$i"
  done
 done
done
echo ALLDONE
