#!/bin/bash
# host_first_touch.sh — run the minimal-first-touch ladder on a GPU HOST, under gpu-lock.
#
#   bash host_first_touch.sh [tag] [timeout_s]
#   MODE=exec CONTAINER=qsa_m455_blyu bash host_first_touch.sh        # into an existing container
#   QSA_ROOT=/home/blyu/qsa IMG=rocm/...20261015 bash host_first_touch.sh
#   DMESG=1 bash host_first_touch.sh                                   # also print amdgpu faults
#
# SAFETY: this is the *only* GPU step of the window and it is bounded.  If any
# step reports TIMEOUT (or the host stops answering), stop: do not retry, do not
# start another container, capture dmesg, report.  See FIRST_TOUCH.md.
set -uo pipefail
TAG=${1:-ft}
TIMEOUT=${2:-60}
IMG=${IMG:-rocm/fw-bringup-mlperf:gfx1250-atom-20261028-dsv41}
QSA_ROOT=${QSA_ROOT:-/data/blyu/qsa}
MODE=${MODE:-run}
CONTAINER=${CONTAINER:-qsa_m455_blyu}
RES="$QSA_ROOT/results/g2/builder"
G2=/work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2
mkdir -p "$RES" 2>/dev/null

echo "host_first_touch: mode=$MODE img=$IMG root=$QSA_ROOT tag=$TAG timeout=${TIMEOUT}s"
echo "gpu-lock holder: $(gpu-lock --help >/dev/null 2>&1 && echo 'gpu-lock present' || echo 'GPU-LOCK MISSING - STOP')"

if [ "$MODE" = "exec" ]; then
  gpu-lock docker exec -e HIP_VISIBLE_DEVICES=0 "$CONTAINER" \
    bash -lc "bash $G2/first_touch.sh $TIMEOUT $TAG"
  RC=$?
else
  G_VIDEO=$(getent group video | cut -d: -f3)
  G_RENDER=$(getent group render | cut -d: -f3)
  gpu-lock docker run --rm --ipc=host --device=/dev/kfd --device=/dev/dri \
    --group-add "${G_VIDEO:-0}" --group-add "${G_RENDER:-0}" \
    -v "$QSA_ROOT:/work" -w /work/work/qsa-v0 \
    -e PYTHONPATH=/work/pylibs:/work/work/qsa-v0 \
    -e QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline \
    -e QSA_RESULTS=/work/results/g2/builder \
    -e FLYDSL_RUNTIME_CACHE_DIR=/work/results/g2/builder/flydsl_cache_"$TAG" \
    --entrypoint bash "$IMG" -c "bash $G2/first_touch.sh $TIMEOUT $TAG"
  RC=$?
fi

echo "host_first_touch: rc=$RC"
if [ "${DMESG:-0}" = "1" ]; then
  echo "== dmesg (amdgpu) =="
  dmesg 2>/dev/null | grep -iE "amdgpu|VM_PAGE|GPU Hang|ring .* timeout" | tail -20 \
    || echo "(dmesg not readable without privileges - capture it from the host console)"
fi
exit "$RC"
