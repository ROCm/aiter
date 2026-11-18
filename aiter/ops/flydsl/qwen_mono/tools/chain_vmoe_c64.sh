#!/bin/bash
# 11-16: c64, the settings of m2_c64_mtp3_B (11-13: interval 8, six decode flags, official ruleset, drafter kernel, 10-minute cap,
# gdn1112 trees) + ATOM_VMOE=1 (the verifier's MoE block as one dataflow kernel on decode steps of <= 16 rows).
S=/runs/chain_vmoe_c64.status
K="ATOM_UA_DECODE_TOKEN_SPLIT=1 ATOM_MTP_ADVANCE_TOKEN_SPLIT=1 ATOM_UA_SPLIT_REDUCE_ONESHOT=1 ATOM_MTP_FUSED_VERIFY_EPILOGUE=1 ATOM_MTP_VERIFY_HOST_HANDOFF=1 ATOM_SAMPLER_GLUON_SELECT=1 SUBMISSION_RULESET=mlperf-endpoints-v1.0-2026-10-C1-A INTERVAL=8"
V="ATOM_VMOE=1 ATOM_VMOE_DIR=/work/mono/dk ATOM_VMOE_CKPT=/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead"
echo "START $(date -u +%FT%TZ)" >> $S
for spec in "$@"; do
  set -- $spec; n=$1; shift
  echo "$n start $(date -u +%FT%TZ) shm free $(df -h /dev/shm | awk 'NR==2{print $4}') $*" >> $S
  /runs/mono_run.sh $n 64 $K ATOM_MTP_MONO=1 "$@"
  echo "$n $(grep -E '^(DONE|DIED|FAIL|SERVER_DIED|CFG_FAIL)' /runs/$n/status.txt | tail -1) | $(grep -h OUTPUTS /runs/$n/status.txt | sed 's/.*salad-like[^:]*: /salad /')" >> $S
  sleep 20
done
echo CHAIN_DONE $(date -u +%FT%TZ) >> $S
