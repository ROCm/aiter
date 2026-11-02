#!/bin/bash
set -euo pipefail
# Batch-specific recipe validated by final-micro-*.json; decode only.
export ATOM_DSV4_EDGE_MASK=0 ATOM_DSV4_TILE_MASK=192 ATOM_DSV4_OVERLAP=2
export ATOM_DSV4_TILE_MASK_C1=128 ATOM_DSV4_OVERLAP_C1=0
export ATOM_DSV4_TILE_MASK_C2=128 ATOM_DSV4_OVERLAP_C2=0
export ATOM_DSV4_BATCH_FULL_CHECK=0
exec bash /work/integrated-c8/run.sh "${@:-fused}"
