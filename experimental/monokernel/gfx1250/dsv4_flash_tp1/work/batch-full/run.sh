#!/bin/bash
set -euo pipefail
mode=${1:?baseline|fused|client}
if [ "$mode" = client ]; then
 exec bash /work/concurrency-rollout/clients.sh "${2:?tag}" "${HTTP_PORT:-8000}" "${3:?concurrency}"
fi
unset ATOM_DSV4_BATCH_QKV ATOM_DSV4_BATCH_QA ATOM_DSV4_DUAL_REGION ATOM_DSV4_BATCH_SERIAL ATOM_DSV4_NATIVE_STATE ATOM_DSV4_MODELWIDE_VALIDATE
export ATOM_DSV4_FULL_LAYER_EXPERIMENTAL=0 ATOM_DSV4_MODELWIDE=0 ATOM_DSV4_BATCH_FULL=0
if [ "$mode" = fused ]; then
 export ATOM_DSV4_FULL_LAYER_EXPERIMENTAL=1 ATOM_DSV4_MODELWIDE=1 ATOM_DSV4_BATCH_FULL=1 ATOM_DSV4_MODELWIDE_MAX_CONTEXT=2048
 export ATOM_DSV4_MATCHED_ATTN=1 ATOM_DSV4_PRECISE_MHC=1 ATOM_DSV4_STATE_DISPATCH=1
elif [ "$mode" != baseline ]; then exit 2; fi
exec bash /work/concurrency-rollout/server.sh
