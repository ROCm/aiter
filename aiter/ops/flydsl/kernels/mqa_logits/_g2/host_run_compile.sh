#!/bin/bash
# host_run_compile.sh <tag>  — runs on the a07-3 HOST.
# Starts a container with NO GPU devices, so the compile-only check cannot touch
# the accelerator (deliberate: the node reported a GPU hang during this session).
set -uo pipefail
TAG=${1:-compile0}
IMG=rocm/fw-bringup-mlperf:gfx1250-atom-20261028-dsv41
G2=/work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2
docker run --rm -v /data/blyu/qsa:/work -w /work/work/qsa-v0 \
  -e PYTHONPATH=/work/pylibs:/work/work/qsa-v0 \
  -e QSA_RESULTS=/work/results/g2/builder \
  -e COMPILE_ONLY=1 \
  -e FLYDSL_GPU_ARCH=gfx1250 \
  -e FLYDSL_RUNTIME_CACHE_DIR=/work/results/g2/builder/flydsl_cache_compile_"$TAG" \
  --entrypoint bash "$IMG" -c "mkdir -p \$FLYDSL_RUNTIME_CACHE_DIR; python3 $G2/compile_only.py 2>&1 | tail -60"
exit $?
