#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ulimit -c 0
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/triton-cache}"
python validate_final.py
python final_bench.py
python validate_layers.py
python layer_sequence_bench.py
