#!/usr/bin/env bash
# Prototype vLLM launcher for the K3 latent FHMoE overlay.
#
# The integrated FlyDSL path is numerically correct in eager execution. CUDA
# graph replay is not yet safe: every TP rank faults on the first M=8 replay
# after a long prefill. Keep --enforce-eager until that is fixed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(
  python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))'
)"
python3 "$SCRIPT_DIR/k3_latent_vllm_overlay.py" "$VLLM_ROOT"
export VLLM_ROCM_USE_K3_LATENT_FHMOE=1
export VLLM_ROCM_K3_LATENT_STRICT=1
export VLLM_ROCM_K3_LATENT_SEPARATED_LAYOUT=1
# Keep the fallback and latent path on one physical GGUU weight layout.
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=0
export AITER_SITUV2_A8W4=0
exec /usr/local/bin/vllm "$@" --enforce-eager
