#!/bin/bash
# ROUND-NATIVE64 host staging entrypoint. NO devices, NO GPU workloads.
# Usage: host_native64_cpu.sh STAGE {cpu|isa}; STAGE/{base,final}/source are git archives.
set -euo pipefail
STAGE=$(realpath "${1:?staging directory}")
MODE=${2:?cpu or isa}
IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-20261015
GOLDEN=/data/blyu/qsa/task/MI455_QSA_INDEX_SCORER/golden_baseline
RESULTS=$STAGE/results-$MODE
mkdir -p "$RESULTS"
docker image inspect "$IMAGE" --format '{{.Id}}' > "$RESULTS/image-id.txt"
printf 'stage=%s\nimage=%s\nmode=%s\nnetwork=none\ndevices=none\n' "$STAGE" "$IMAGE" "$MODE" > "$RESULTS/host-procedure.txt"
docker run --rm --network none --name "qsa-native64-$MODE-$(date +%s)" \
 -v "$STAGE/final/source:/source:ro" -v "$STAGE/base/source:/base:ro" -v "$RESULTS:/results" \
 -v "$GOLDEN:/work/task/MI455_QSA_INDEX_SCORER/golden_baseline:ro" -w /source \
 -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
 -e QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline \
 --entrypoint bash "$IMAGE" /source/aiter/ops/flydsl/kernels/mqa_logits/_g2/container_native64_cpu.sh "$MODE"
