#!/bin/bash
# ROUND-PAD2 host staging entrypoint. NO devices, NO GPU workloads.
set -euo pipefail
STAGE=$(realpath "${1:?staging directory}")
PHASE=${2:?base or final or cpu}
IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-20261015
GOLDEN=/data/blyu/qsa/task/MI455_QSA_INDEX_SCORER/golden_baseline
SOURCE=$STAGE/$PHASE/source
MODE=$PHASE
if [[ $PHASE == cpu ]]; then SOURCE=$STAGE/final/source; MODE=cpu; fi
if [[ $PHASE == cpu2 ]]; then SOURCE=$STAGE/final2/source; MODE=cpu; fi
if [[ $PHASE == cpu3 ]]; then SOURCE=$STAGE/final3/source; MODE=cpu; fi
if [[ $PHASE == native ]]; then SOURCE=$STAGE/final3/source; MODE=native; fi
RESULTS=$STAGE/$PHASE/results
mkdir -p "$RESULTS"
docker image inspect "$IMAGE" > "$RESULTS/image-inspect.json"
docker version > "$RESULTS/docker-version.txt"
printf 'source=%s\nresults=%s\ngolden=%s\nimage=%s\nphase=%s\nnetwork=none\ndevices=none\n' "$SOURCE" "$RESULTS" "$GOLDEN" "$IMAGE" "$PHASE" > "$RESULTS/host-procedure.txt"
sha256sum "$STAGE/tools"/* > "$RESULTS/tools.sha256"
docker run --rm --network none --name "qsa-pad2-$PHASE-$(date +%s)" \
 -v "$SOURCE:/source:ro" -v "$STAGE/base/source:/base:ro" -v "$STAGE/tools:/tools:ro" -v "$RESULTS:/results" \
 -v "$GOLDEN:/work/task/MI455_QSA_INDEX_SCORER/golden_baseline:ro" -w /source \
 -e COMPILE_ONLY=1 -e FLYDSL_GPU_ARCH=gfx1250 -e PYTHONDONTWRITEBYTECODE=1 \
 -e QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline \
 --entrypoint bash "$IMAGE" /tools/container_pad2_cpu_isa.sh "$MODE"
