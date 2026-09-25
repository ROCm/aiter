#!/bin/bash
# Build the patched-Triton image used to reproduce this PR's numbers.
#
#   ./build.sh                       # default tag
#   IMAGE_TAG=my/tag:v1 ./build.sh   # override
#
# Takes ~10 minutes after the base image pull: Triton is compiled from source.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE_TAG="${IMAGE_TAG:-aiter-blasst/triton-chaindot:rocm7.2.4-py3.12-torch2.10.0}"

# Build context is patch/, not docker/ -- the Dockerfile COPYs the patch, and a
# context of docker/ would not contain it.
docker build \
  -f "${HERE}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  "${HERE}/../patch"

echo
echo "Built: ${IMAGE_TAG}"
echo "Verify inside the container:"
echo "  cat /opt/triton-src-commit.txt   # pinned commit + patch applied"
echo "  python3 -c 'import triton; print(triton.__version__)'"
