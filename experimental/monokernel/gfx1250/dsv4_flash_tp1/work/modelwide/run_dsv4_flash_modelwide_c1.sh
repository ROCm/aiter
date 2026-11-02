#!/bin/bash
set -euo pipefail
mode=${1:?Usage: $0 baseline|fused|matched|client-baseline|client-fused|client-matched}
case "$mode" in
 baseline)
  export ATOM_DSV4_FULL_LAYER_EXPERIMENTAL=0 ATOM_DSV4_MODELWIDE=0
  bash /work/modelwide/serve.sh
  ;;
 fused)
  export ATOM_DSV4_FULL_LAYER_EXPERIMENTAL=1 ATOM_DSV4_MODELWIDE=1 ATOM_DSV4_MODELWIDE_MAX_CONTEXT=2048
  bash /work/modelwide/serve.sh
  ;;
 matched)
  export ATOM_DSV4_FULL_LAYER_EXPERIMENTAL=1 ATOM_DSV4_MODELWIDE=1 ATOM_DSV4_MODELWIDE_MAX_CONTEXT=2048 ATOM_DSV4_MATCHED_ATTN=1
  bash /work/modelwide/serve.sh
  ;;
 client-baseline|client-fused|client-matched)
  bash /work/modelwide/clients.sh "${mode#client-}" "${HTTP_PORT:-8000}"
  ;;
 *) exit 2;;
esac
