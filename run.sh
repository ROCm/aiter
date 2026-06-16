#!/bin/bash
set -euo pipefail

TOKENS="${1:-1}"
MODEL="${2:-all}"
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

declare -A MODELS
MODELS[gpt]="--experts 128 --tokens $TOKENS --topk 4 --model-dim 3072 --inter-dim 3072"
MODELS[kimi]="--experts 384 --tokens $TOKENS --topk 8 --model-dim 7168 --inter-dim 2048"
MODELS[dsr1]="--experts 256 --tokens $TOKENS --topk 8 --model-dim 7168 --inter-dim 2048"
MODELS[dsv4-pro]="--experts 384 --tokens $TOKENS --topk 6 --model-dim 7168 --inter-dim 3072"
MODELS[dsv4-flash]="--experts 256 --tokens $TOKENS --topk 6 --model-dim 4096 --inter-dim 2048"

run_bench() {
    local name=$1
    local args=${MODELS[$name]}
    local logfile="${name}_${BRANCH}_t${TOKENS}.log"
    echo "=== $name | branch=$BRANCH tokens=$TOKENS ==="
    AITER_LOG_MORE=1 AITER_FORCE_A8W4=1 AITER_USE_GROUPED_GEMM=1 AITER_FORCE_GFX1250=1 \
        python op_tests/test_flydsl_grouped_gemm_gfx1250.py \
        --scenario bench --data-format a8w4 --layout gguu \
        $args --act silu --no-bias 2>&1 | tee "$logfile"
}

if [ "$MODEL" = "all" ]; then
    for name in gpt kimi dsv4-pro dsv4-flash; do
        run_bench "$name"
    done
else
    if [[ -z "${MODELS[$MODEL]+x}" ]]; then
        echo "Unknown model: $MODEL. Choose from: gpt kimi dsv4-pro dsv4-flash (or all)"
        exit 1
    fi
    run_bench "$MODEL"
fi
