#!/bin/bash
# Experimental Flash layer-0 monokernel. Numerical validation is not accepted.
# One server start, then run clients.sh twice via its built-in two-round loop.
set -euo pipefail
export ATOM_DSV4_FULL_LAYER_ROOT="${ATOM_DSV4_FULL_LAYER_ROOT:-/work/full-layer-native}"
export ATOM_DSV4_FULL_LAYER_EXPERIMENTAL="${ATOM_DSV4_FULL_LAYER_EXPERIMENTAL:-1}"
export ATOM_DSV4_HYBRID=0
export ATOM_DSV4_FLASH_MOE=0
export PYTHONPATH="/app/aiter:/app/ATOM:${PYTHONPATH:-}"
exec bash "$ATOM_DSV4_FULL_LAYER_ROOT/serve.sh" "${1:-/data/DeepSeek-V4-Flash}"
