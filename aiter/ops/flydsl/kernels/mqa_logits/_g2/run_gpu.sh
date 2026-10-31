#!/bin/bash
# GPU correctness / first-look driver (runs INSIDE the container under gpu-lock).
#   bash run_gpu.sh <stage> <tag>
set -uo pipefail
STAGE=${1:-correctness}
TAG=${2:-run}
export PYTHONPATH=/work/pylibs:/work/work/qsa-v0
export QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline
export QSA_RESULTS=${QSA_RESULTS:-/work/results/g2/builder}
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-/work/results/g2/builder/flydsl_cache}
mkdir -p "$QSA_RESULTS" "$FLYDSL_RUNTIME_CACHE_DIR"
cd /work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2
python3 run_gpu.py --stage "$STAGE" --tag "$TAG" 2>&1 | tee "$QSA_RESULTS/gpu_${TAG}.log"
exit "${PIPESTATUS[0]}"
