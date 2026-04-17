#!/usr/bin/env bash
# =============================================================================
# run_atom.sh — Run ATOM GSM8K accuracy test on AITER runners
#
# Usage:
#   run_atom.sh <aiter_sha> <model_path> <threshold> [extra_args] [aiter_index_url]
#
# Required env:
#   HF_TOKEN — HuggingFace token
#
# Optional env:
#   ATOM_BASE_IMAGE — default: rocm/atom-dev:latest
#   ATOM_BRANCH     — default: main
#   ATOM_REPO       — default: ROCm/ATOM
#   ATOM_ENV_VARS   — extra env vars for container
# =============================================================================
set -euo pipefail

CONTAINER_NAME="atom_nightly_$$"
cleanup() { docker stop "${CONTAINER_NAME}" 2>/dev/null || true; docker rm "${CONTAINER_NAME}" 2>/dev/null || true; }
trap cleanup EXIT

AITER_SHA="${1:?Usage: run_atom.sh <aiter_sha> <model_path> <threshold> [extra_args] [aiter_index_url]}"
MODEL_PATH="${2:?model_path required}"
THRESHOLD="${3:?threshold required}"
EXTRA_ARGS="${4:-}"
AITER_INDEX_URL="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve_aiter_version.sh"

ATOM_BASE_IMAGE="${ATOM_BASE_IMAGE:-rocm/atom-dev:latest}"
ATOM_BRANCH="${ATOM_BRANCH:-main}"
ATOM_REPO="${ATOM_REPO:-ROCm/ATOM}"
SHORT_SHA="${AITER_SHA:0:7}"
WORKDIR="${GITHUB_WORKSPACE:-.}"

echo "=== ATOM Accuracy Test ==="
echo "AITER SHA:   ${AITER_SHA}"
echo "Model:       ${MODEL_PATH}"
echo "Threshold:   ${THRESHOLD}"
echo "Extra args:  ${EXTRA_ARGS}"
echo ""

# ── Checkout ATOM into workspace ──
ATOM_DIR="${WORKDIR}/atom-checkout"
rm -rf "${ATOM_DIR}"
git clone --depth 1 -b "${ATOM_BRANCH}" "https://github.com/${ATOM_REPO}.git" "${ATOM_DIR}"

# ── Build image ──
docker pull "${ATOM_BASE_IMAGE}"

cat > "${ATOM_DIR}/Dockerfile.nightly" <<EOF
FROM ${ATOM_BASE_IMAGE}
RUN pip uninstall -y amd-aiter || true
RUN pip install --upgrade "pybind11>=3.0.1"
EOF

if [ -n "${AITER_INDEX_URL}" ]; then
  echo "RUN ${AITER_INSTALL_CMD}" >> "${ATOM_DIR}/Dockerfile.nightly"
else
  cat >> "${ATOM_DIR}/Dockerfile.nightly" <<EOF
RUN git clone https://github.com/ROCm/aiter.git /app/aiter-test && \
    cd /app/aiter-test && git checkout ${AITER_SHA} && \
    git submodule sync && git submodule update --init --recursive && \
    MAX_JOBS=64 PREBUILD_KERNELS=0 GPU_ARCHS=gfx950 pip install -e .
EOF
fi

echo 'RUN echo "=== AITER version ===" && pip show amd-aiter || true' >> "${ATOM_DIR}/Dockerfile.nightly"

docker build --network=host --no-cache \
  -t "rocm/atom-aiter-ci:${SHORT_SHA}" \
  -f "${ATOM_DIR}/Dockerfile.nightly" "${ATOM_DIR}"

# ── GPU devices ──
if [ -f "/etc/podinfo/gha-render-devices" ]; then
  DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
else
  DEVICE_FLAG="--device /dev/dri"
fi

# ── Model mount ──
MODEL_MOUNT=""
if [ -d "/models" ]; then
  MODEL_MOUNT="-v /models:/models"
fi

# ── Env vars ──
ENV_FILE_FLAG=""
if [ -n "${ATOM_ENV_VARS:-}" ]; then
  echo "${ATOM_ENV_VARS}" > /tmp/atom_env_${$}.txt
  ENV_FILE_FLAG="--env-file /tmp/atom_env_${$}.txt"
fi

# ── Start container (mount ATOM checkout as /workspace) ──
docker ps -aq -f name="${CONTAINER_NAME}" | xargs -r docker stop | xargs -r docker rm || true

docker run -dt --device=/dev/kfd ${DEVICE_FLAG} \
  ${MODEL_MOUNT} ${ENV_FILE_FLAG} \
  --ipc=host --group-add video \
  --shm-size=16G --privileged \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e ATOM_DISABLE_MMAP=true \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${ATOM_DIR}:/workspace" \
  -w /workspace \
  --name "${CONTAINER_NAME}" \
  "rocm/atom-aiter-ci:${SHORT_SHA}"

# ── Check versions ──
docker exec "${CONTAINER_NAME}" bash -lc "pip show amd-aiter atom && pip list" || true

# ── Resolve model path (use cache if available) ──
model_cache="/models"
model_arg="${MODEL_PATH}"
if docker exec "${CONTAINER_NAME}" bash -lc "[ -d '${model_cache}/${MODEL_PATH}' ]" 2>/dev/null; then
  model_arg="${model_cache}/${MODEL_PATH}"
  echo "Using cached model: ${model_arg}"
else
  echo "Model not cached, will download: ${MODEL_PATH}"
fi

# ── Run test ──
MODEL_NAME=$(echo "${MODEL_PATH}" | sed 's/\//_/g')
LOG_FILE="${WORKDIR}/atom_nightly_${MODEL_NAME}.txt"

echo ""
echo "=== Launching ATOM server ==="
docker exec "${CONTAINER_NAME}" bash -lc \
  ".github/scripts/atom_test.sh launch ${model_arg} ${EXTRA_ARGS}" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "=== Running accuracy test ==="
docker exec "${CONTAINER_NAME}" bash -lc \
  ".github/scripts/atom_test.sh accuracy ${model_arg}" 2>&1 | tee -a "${LOG_FILE}"

# ── Check threshold ──
# Results are in the mounted workspace
result_file=$(ls -1t "${ATOM_DIR}/accuracy_test_results/"*.json 2>/dev/null | head -n 1 || true)
if [ -z "$result_file" ] || [ ! -f "$result_file" ]; then
  echo "ERROR: No results JSON found in ${ATOM_DIR}/accuracy_test_results/"
  exit 2
fi

score=$(jq '.results.gsm8k["exact_match,flexible-extract"]' "$result_file")
echo ""
echo "=== Result ==="
echo "GSM8K score: ${score} (threshold: ${THRESHOLD})"

failed=$(awk -v val="$score" -v thr="${THRESHOLD}" 'BEGIN {print (val < thr) ? 1 : 0}')
if [ "$failed" -eq 1 ]; then
  echo "FAILED: ${score} < ${THRESHOLD}"
  exit 1
fi
echo "PASSED"
