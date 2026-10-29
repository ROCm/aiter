#!/usr/bin/env bash
# Build the patched Triton (ROCm/triton 669b31ac + rank3_mfma_scaled.patch) as a
# wheel inside the vLLM nightly image (same Python / ABI), then the image the
# Gluon port runs in (FROM the nightly, the wheel installed).
# Usage: WORK=/scratch/triton ./build.sh   (clones ROCm/triton into $WORK/src)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HOME/triton-mfma-scaled}
BASE=${BASE:-vllm/vllm-openai-rocm:nightly-rocm100-00b6c4c565a8732c131e1f452d4a17803bb0842b}
TAG=${TAG:-vllm-nightly-tritonfix:00b6c4c5}
mkdir -p "$WORK/wheels"
if [ ! -d "$WORK/src" ]; then
  git clone https://github.com/ROCm/triton.git "$WORK/src"
  git -C "$WORK/src" checkout 669b31acc1dd1b3fd93286afd5db67f65d9f7557
  git -C "$WORK/src" apply "$HERE/rank3_mfma_scaled.patch"
fi
docker run --rm -v "$WORK":/w -w /w/src -e MAX_JOBS=${MAX_JOBS:-96} --entrypoint bash "$BASE" -c '
set -e
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
git config --global --add safe.directory /w/src
pip install -q "cmake>=3.20,<4.0" nanobind==2.10.2 pybind11 ninja
pip wheel --no-build-isolation --no-deps -w /w/wheels . 2>&1 | tail -20
rc=${PIPESTATUS[0]}
chown -R '"$(id -u):$(id -g)"' /w
exit $rc'
cp "$HERE/Dockerfile" "$WORK/Dockerfile"
docker build --build-arg BASE="$BASE" -t "$TAG" "$WORK"
