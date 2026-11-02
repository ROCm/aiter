#!/bin/bash
set -euo pipefail
export ATOM_DSV4_DECODE_ONLY_DISPATCH=1
exec bash /work/integrated-c8/run.sh "${1:?baseline|fused|client}" "${@:2}"
