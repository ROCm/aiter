#!/bin/bash
# ROUND-MERGE1 host staging entrypoint. NO devices, NO GPU workloads.
# Usage: host_merge1_cpu.sh STAGE {cpu-default|cpu-optout|isa}
# STAGE/{final,base,draft}/source = git archives of merge / NATIVE64 60b9aade / DRAFT f6a54cc9.
set -euo pipefail
STAGE=$(realpath "${1:?staging directory}")
MODE=${2:?cpu-default, cpu-optout or isa}
IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-20261015
GOLDEN=/data/blyu/qsa/task/MI455_QSA_INDEX_SCORER/golden_baseline
RESULTS=$STAGE/results
mkdir -p "$RESULTS/$MODE"; chmod a+rwx "$RESULTS" "$RESULTS/$MODE" 2>/dev/null || true
docker image inspect "$IMAGE" --format '{{.Id}}' > "$RESULTS/$MODE/image-id.txt"
printf 'stage=%s\nimage=%s\nmode=%s\nnetwork=none\ndevices=none\n' "$STAGE" "$IMAGE" "$MODE" > "$RESULTS/$MODE/host-procedure.txt"
docker run --rm --network none --name "qsa-merge1-$MODE-$(date +%s)" \
 -v "$STAGE/final/source:/source:ro" -v "$STAGE/base/source:/base:ro" -v "$STAGE/draft/source:/draft:ro" \
 -v "$RESULTS:/results" \
 -v "$GOLDEN:/work/task/MI455_QSA_INDEX_SCORER/golden_baseline:ro" -w /source \
 -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
 -e QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline -e ISA_JOBS="${ISA_JOBS:-16}" \
 --entrypoint bash "$IMAGE" /source/aiter/ops/flydsl/kernels/mqa_logits/_g2/container_merge1_cpu.sh "$MODE"
