#!/usr/bin/env bash
# Confirm AITER_TDM_WIDE_KSL=1 no longer forces gemm2 (KWS=1) into the wide path.
# Runs WIDE=1 then WIDE unset, same batch, so the numbers are comparable.
set -u
cd /data/yanguahe/code/wk_sp1/aiter || exit 1

for mode in wide1 unset; do
  echo "############ $mode ############"
  if [ "$mode" = wide1 ]; then
    export AITER_TDM_WIDE_KSL=1
  else
    unset AITER_TDM_WIDE_KSL
  fi
  bash my_code/sweep_tdm.sh g2_m64_nb3 2>&1
  cp my_code/sweep_tdm/g2_m64_nb3.log my_code/sweep_tdm/verify_$mode.log
done

echo "############ sched lines ############"
for mode in wide1 unset; do
  echo "--- $mode ---"
  grep '\[sched\]' my_code/sweep_tdm/verify_$mode.log
  grep -o 'rel_l2=[0-9.e+-]*' my_code/sweep_tdm/verify_$mode.log | tail -1
done
