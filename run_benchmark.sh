#!/bin/bash

IMAGE=""
WORKDIR="/work"

DOCKER_CMD="\
  git config --global --add safe.directory '*' && \
  git submodule update --init --recursive > /dev/null 2>&1 && \
  python3 setup.py develop > /dev/null 2>&1 && \
  python3 benchmark_gemm.py \
"

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  --shm-size 16G \
  -v $(pwd):${WORKDIR} \
  -w ${WORKDIR} \
  ${IMAGE} \
  bash -c "${DOCKER_CMD}"
