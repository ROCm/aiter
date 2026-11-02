#!/bin/bash
set -euo pipefail
name=${1:?native|previous|selected|replica}
while [ ! -f /work/rollout-c1248/micro-approved ] && [ ! -f /work/rollout-c1248/startup-${name} ]; do sleep 2; done
export PYTHONPATH=/work/rollout-c1248:/app/aiter:/app/ATOM
export ATOM_MK_PREFILL_AUDIT=/work/rollout-c1248/prefill-${name}
export SERVER_LOG=/work/rollout-c1248/${name}-server.log
export CONCURRENCY=8 CAPTURE_SIZES='[1,2,4,8]'
export ATOM_DSV4_BATCH_FULL_CHECK=0 ATOM_DSV4_EDGE_MASK=0 ATOM_DSV4_TILE_MASK=0 ATOM_DSV4_OVERLAP=0
export ATOM_DSV4_HYBRID=0 ATOM_DSV4_HYBRID_GEMM=0
case "$name" in
 native) mode=baseline;;
 previous) mode=fused;;
 selected|replica) mode=fused; export ATOM_DSV4_TILE_MASK=192 ATOM_DSV4_OVERLAP=2 ATOM_DSV4_TILE_MASK_C1=128 ATOM_DSV4_OVERLAP_C1=0 ATOM_DSV4_TILE_MASK_C2=128 ATOM_DSV4_OVERLAP_C2=0;;
 *) exit 2;;
esac
exec bash /work/integrated-c8/run.sh "$mode"
