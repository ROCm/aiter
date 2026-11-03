#!/bin/bash
# Host entrypoint (NO devices, NO GPU workloads): runs container_cpu.sh in the pinned image.
# Usage: host_cpu.sh STAGE {cpu|isa}
#   STAGE/final/source = git archive of qsa-index-scorer/stage2, STAGE/base/source = champion a78c285c.
set -euo pipefail
STAGE=$(realpath "${1:?staging directory}")
MODE=${2:?cpu or isa}
IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-20261015
GOLDEN=${GOLDEN:-/data/blyu/qsa/task/MI455_QSA_INDEX_SCORER/golden_baseline}
RESULTS=$STAGE/results
mkdir -p "$RESULTS/$MODE"; chmod a+rwx "$RESULTS" "$RESULTS/$MODE" 2>/dev/null || true
docker image inspect "$IMAGE" --format '{{.Id}}' > "$RESULTS/$MODE/image-id.txt"
printf 'stage=%s\nimage=%s\nmode=%s\nnetwork=none\ndevices=none\n' "$STAGE" "$IMAGE" "$MODE" \
  > "$RESULTS/$MODE/host-procedure.txt"
docker run --rm --network none --name "qsa-stage2-$MODE-$(date +%s)" \
  -v "$STAGE/final/source:/source:ro" -v "$STAGE/base/source:/base:ro" -v "$RESULTS:/results" \
  -v "$GOLDEN:/work/task/MI455_QSA_INDEX_SCORER/golden_baseline:ro" -w /source \
  -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
  -e QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline -e ISA_JOBS="${ISA_JOBS:-16}" \
  --entrypoint bash "$IMAGE" /source/op_tests/op_benchmarks/qsa_index_scorer/container_cpu.sh "$MODE"
