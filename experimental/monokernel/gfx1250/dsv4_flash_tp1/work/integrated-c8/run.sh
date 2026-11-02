#!/bin/bash
set -euo pipefail
export ATOM_DSV4_DECODE_ONLY_DISPATCH=${ATOM_DSV4_DECODE_ONLY_DISPATCH:-1}
export ATOM_DSV4_BATCH_FULL_DIR=/work/integrated-c8
exec bash /work/batch-full/run.sh "${1:?baseline|fused|client}" "${@:2}"
