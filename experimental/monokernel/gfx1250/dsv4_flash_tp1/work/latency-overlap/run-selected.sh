#!/bin/bash
set -euo pipefail
# Validated full-layer layout fusion and early OA/OB TDM; experimental tile semaphores off.
export ATOM_DSV4_EDGE_MASK=0 ATOM_DSV4_TILE_MASK=192 ATOM_DSV4_OVERLAP=2
export ATOM_DSV4_BATCH_FULL_CHECK=0
exec bash /work/integrated-c8/run.sh "${@:-fused}"
