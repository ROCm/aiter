#!/bin/bash
# 11-16: 4-minute-cap c64 points, m2_c64_mtp3_B settings + ATOM_VMOE with one diagnostic setting each ("name K=V ..." arguments).
S=/runs/chain_vmoe_diag.status
K="ATOM_UA_DECODE_TOKEN_SPLIT=1 ATOM_MTP_ADVANCE_TOKEN_SPLIT=1 ATOM_UA_SPLIT_REDUCE_ONESHOT=1 ATOM_MTP_FUSED_VERIFY_EPILOGUE=1 ATOM_MTP_VERIFY_HOST_HANDOFF=1 ATOM_SAMPLER_GLUON_SELECT=1 SUBMISSION_RULESET=mlperf-endpoints-v1.0-2026-10-C1-A INTERVAL=8"
V="ATOM_MTP_MONO_DIR=/work/mono/dk ATOM_MTP_MONO_HEAD=/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead ATOM_MTP_MONO=1 ATOM_VMOE=1 ATOM_VMOE_DIR=/work/mono/dk ATOM_VMOE_CKPT=/data/Qwen3.6-35B-A3B-MXFP4"
ulimit -c 0
for spec in "$@"; do
  set -- $spec; n=$1; shift
  echo "$n start $(date -u +%FT%TZ) $*" >> $S
  EXTRA="$K $V $*" AITER_DIR=/work/push/gdn1112_aiter R=/work/qwen35_ep CAP_MIN=4 NUM_TRAJ=1226 /work/push/gdn1112_atom/recipes/mlperf_endpoints_qwen35/run_point_20261107.sh 64 /runs/$n > /runs/$n.run.log 2>&1
  echo "$n $(grep -E '^(DONE|DIED|FAIL|SERVER_DIED|CFG_FAIL)' /runs/$n/status.txt | tail -1 | cut -c1-40) | $(grep -h 'MTP Stats  ' /runs/$n/c64/server.log | tail -1 | sed 's/.*Acceptance rate: \([0-9.]*%\).*/acceptance \1/') | $(grep -h OUTPUTS /runs/$n/status.txt | sed 's/.*salad-like[^:]*: /salad /')" >> $S
  sleep 10
done
echo CHAIN_DONE $(date -u +%FT%TZ) >> $S
