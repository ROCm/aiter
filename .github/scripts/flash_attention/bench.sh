#!/bin/bash
# Run the Flash Attention benchmarks of aiter CI. CI and a local container run the same lines.
#
# Usage: bench.sh <fa | mha> [out-dir]
#   fa    FlashAttention's own benchmark on its Triton route
#   mha   aiter's MHA benchmark of the Dao-AI implementation, as a CSV in out-dir
#         (default /tmp/bench_dao_ai)
#   Run it in a container that has aiter at /aiter and flash-attention at /flash-attention.
set -u

case "${1:-}" in
fa)
    cd /flash-attention || exit 2
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python benchmarks/benchmark_flash_attention.py
    ;;
mha)
    cd /aiter || exit 2
    python op_tests/op_benchmarks/triton/bench_mha.py \
        -impl dao_ai -metric throughput -o "${2:-/tmp/bench_dao_ai}"
    ;;
*)
    echo "usage: $0 <fa | mha> [out-dir]" >&2
    exit 2
    ;;
esac
