#!/usr/bin/env bash
# =============================================================================
# run_atom.sh — Build ATOM+AITER image and run GSM8K accuracy test
#
# Usage:
#   run_atom.sh <aiter_sha> <model_path> <threshold> [extra_args] [wheel_url]
#
# Required env:
#   HF_TOKEN — HuggingFace token for model access
#
# Optional env:
#   ATOM_BASE_IMAGE — default: rocm/atom-dev:latest
#   ATOM_BRANCH     — default: main
#   ATOM_REPO       — default: ROCm/ATOM
#   ATOM_ENV_VARS   — extra env vars for container (e.g. "ATOM_GPT_OSS_MODEL=1")
# =============================================================================
set -euo pipefail

AITER_SHA="${1:?Usage: run_atom.sh <aiter_sha> <model_path> <threshold> [extra_args] [wheel_url]}"
MODEL_PATH="${2:?model_path required}"
THRESHOLD="${3:?threshold required}"
EXTRA_ARGS="${4:-}"
WHEEL_URL="${5:-}"

ATOM_BASE_IMAGE="${ATOM_BASE_IMAGE:-rocm/atom-dev:latest}"
ATOM_BRANCH="${ATOM_BRANCH:-main}"
ATOM_REPO="${ATOM_REPO:-ROCm/ATOM}"
SHORT_SHA="${AITER_SHA:0:7}"
IMAGE_TAG="rocm/atom-aiter-ci:nightly-${SHORT_SHA}"
CONTAINER_NAME="atom_nightly_$$"

echo "=== ATOM Accuracy Test ==="
echo "AITER SHA:   ${AITER_SHA}"
echo "Model:       ${MODEL_PATH}"
echo "Threshold:   ${THRESHOLD}"
echo "Extra args:  ${EXTRA_ARGS}"
echo "Base image:  ${ATOM_BASE_IMAGE}"
echo ""

# ── Clone ATOM ──
ATOM_WORKSPACE="${RUNNER_TEMP:-/tmp}/atom-checkout"
rm -rf "${ATOM_WORKSPACE}"
git clone --depth 1 -b "${ATOM_BRANCH}" "https://github.com/${ATOM_REPO}.git" "${ATOM_WORKSPACE}"
cd "${ATOM_WORKSPACE}"

# ── Build image ──
cat > Dockerfile.atom-nightly <<EOF
FROM ${ATOM_BASE_IMAGE}
RUN pip uninstall -y amd-aiter || true
RUN pip install --upgrade "pybind11>=3.0.1"
EOF

if [ -n "${WHEEL_URL}" ]; then
  echo "RUN pip install --force-reinstall \"${WHEEL_URL}\"" >> Dockerfile.atom-nightly
else
  cat >> Dockerfile.atom-nightly <<EOF
RUN git clone https://github.com/ROCm/aiter.git /app/aiter-test && \
    cd /app/aiter-test && git checkout ${AITER_SHA} && \
    git submodule sync && git submodule update --init --recursive && \
    MAX_JOBS=64 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 pip install -e .
EOF
fi

echo 'RUN echo "=== AITER version ===" && pip show amd-aiter || true' >> Dockerfile.atom-nightly

docker pull "${ATOM_BASE_IMAGE}"
docker build --network=host --no-cache -t "${IMAGE_TAG}" -f Dockerfile.atom-nightly .

# ── Resolve GPU devices ──
if [ -f "/etc/podinfo/gha-render-devices" ]; then
  DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
else
  DEVICE_FLAG="--device /dev/dri"
fi

# ── Env file ──
ENV_FILE_FLAG=""
if [ -n "${ATOM_ENV_VARS:-}" ]; then
  echo "${ATOM_ENV_VARS}" > /tmp/atom_env.txt
  ENV_FILE_FLAG="--env-file /tmp/atom_env.txt"
fi

# ── Start container ──
docker ps -aq -f name="${CONTAINER_NAME}" | xargs -r docker stop | xargs -r docker rm || true

docker run -dt --device=/dev/kfd ${DEVICE_FLAG} \
  ${ENV_FILE_FLAG} \
  --ipc=host --group-add video \
  --shm-size=16G --privileged \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e ATOM_DISABLE_MMAP=true \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${ATOM_WORKSPACE}:/workspace" \
  -w /workspace \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_TAG}"

# ── Check versions ──
docker exec "${CONTAINER_NAME}" bash -lc "pip show amd-aiter atom && pip list"

# ── Run test ──
MODEL_NAME=$(basename "${MODEL_PATH}")
LOG_FILE="atom_nightly_${MODEL_NAME}.txt"

docker exec "${CONTAINER_NAME}" bash -lc \
  ".github/scripts/atom_test.sh launch ${MODEL_PATH} ${EXTRA_ARGS}"

docker exec "${CONTAINER_NAME}" bash -lc \
  ".github/scripts/atom_test.sh accuracy ${MODEL_PATH}" 2>&1 | tee "${LOG_FILE}"

# ── Check threshold ──
result_file=$(ls -1t accuracy_test_results/*.json 2>/dev/null | head -n 1 || true)
if [ -z "$result_file" ] || [ ! -f "$result_file" ]; then
  echo "ERROR: No results JSON found"
  docker stop "${CONTAINER_NAME}" || true
  docker rm "${CONTAINER_NAME}" || true
  exit 2
fi

score=$(jq '.results.gsm8k["exact_match,flexible-extract"]' "$result_file")
echo ""
echo "=== Result ==="
echo "GSM8K score: ${score} (threshold: ${THRESHOLD})"

failed=$(awk -v val="$score" -v thr="${THRESHOLD}" 'BEGIN {print (val < thr) ? 1 : 0}')

# ── Cleanup ──
docker stop "${CONTAINER_NAME}" || true
docker rm "${CONTAINER_NAME}" || true

if [ "$failed" -eq 1 ]; then
  echo "FAILED: ${score} < ${THRESHOLD}"
  exit 1
fi
echo "PASSED"
