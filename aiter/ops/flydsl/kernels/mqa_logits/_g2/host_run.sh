#!/bin/bash
# host_run.sh <stage> <tag>  — runs on the a07-3 HOST (not in the container).
#   stage: cpu | smoke | correctness | timing | all
# Keeps all quoting inside this file so nothing has to survive an ssh argv round trip.
set -uo pipefail
STAGE=${1:-correctness}
TAG=${2:-run}
IMG=rocm/fw-bringup-mlperf:gfx1250-atom-20261028-dsv41
RES=/data/blyu/qsa/results/g2/builder
G2=/work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2
mkdir -p "$RES"

if [ "$STAGE" = "cpu" ]; then
  docker run --rm -v /data/blyu/qsa:/work -w /work/work/qsa-v0 \
    -e PYTHONPATH=/work/pylibs:/work/work/qsa-v0 \
    -e QSA_RESULTS=/work/results/g2/builder \
    --entrypoint bash "$IMG" -c "bash $G2/run_cpu.sh"
  exit $?
fi

G_VIDEO=$(getent group video | cut -d: -f3)
G_RENDER=$(getent group render | cut -d: -f3)
gpu-lock docker run --rm --ipc=host --device=/dev/kfd --device=/dev/dri \
  --group-add "$G_VIDEO" --group-add "$G_RENDER" \
  -v /data/blyu/qsa:/work -w /work/work/qsa-v0 \
  -e PYTHONPATH=/work/pylibs:/work/work/qsa-v0 \
  -e QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline \
  -e QSA_RESULTS=/work/results/g2/builder \
  -e FLYDSL_RUNTIME_CACHE_DIR=/work/results/g2/builder/flydsl_cache_"$TAG" \
  --entrypoint bash "$IMG" -c "bash $G2/run_gpu.sh $STAGE $TAG"
exit $?
