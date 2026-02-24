#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Run OPUS tests inside ROCm Docker container.
#
# Path mapping (from -v mounts):
#   Host /mnt/raid0     -> Container /raid0
#   Host /home/carhuang -> Container /dockerx
#
# So inside container the repo is at: /raid0/carhuang/repo/aiter
# and this test dir is:              /raid0/carhuang/repo/aiter/op_tests/opus
#
# Usage:
#   ./run_tests_in_docker.sh              # run full test suite, then exit
#   ./run_tests_in_docker.sh bash         # interactive shell in container (cd to opus dir)
#   ./run_tests_in_docker.sh ./run_tests.sh      # run full test suite inside container

set -e

IMAGE="${OPUS_DOCKER_IMAGE:-rocm/atom:nightly_202601190317}"

# Path inside the container (do not use host path /mnt/raid0 here)
OPUS_DIR_IN_DOCKER="/raid0/carhuang/repo/aiter/op_tests/opus"

DOCKER_ARGS=(
  --privileged
  --network=host
  --device=/dev/kfd
  --device=/dev/dri
  --group-add=video
  -v /home/carhuang:/dockerx
  -v /mnt/raid0:/raid0
)

if [[ $# -eq 0 ]]; then
  docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c "cd $OPUS_DIR_IN_DOCKER && ./run_tests.sh"
else
  if [[ "$*" == "bash" ]] || [[ "$*" == "sh" ]]; then
    echo "Starting shell in container. Repo at /raid0/carhuang/repo/aiter, op_tests/opus at $OPUS_DIR_IN_DOCKER"
    docker run -it "${DOCKER_ARGS[@]}" "$IMAGE" bash -c "cd $OPUS_DIR_IN_DOCKER && exec bash"
  else
    docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c "cd $OPUS_DIR_IN_DOCKER && $*"
  fi
fi
