#!/usr/bin/env bash
# =============================================================================
# run_afo_llm.sh — Run AFO-LLM GSM8K accuracy test via vLLM + lm_eval
#
# Usage:
#   run_afo_llm.sh <aiter_sha> <model_key> <quantizer> <tp> <threshold> \
#                   [extra_args] [aiter_index_url]
#
# Required env:
#   HF_TOKEN — HuggingFace token
#
# Optional env:
#   VLLM_BASE_IMAGE   — default: from vllm_pins.json or rocm/vllm-dev:nightly
#   AFO_LLM_BRANCH    — default: main
#   AFO_LLM_REPO      — default: ROCm/AFO-LLM
#   AFO_LLM_ENV_VARS  — extra env vars for container
# =============================================================================
set -euo pipefail

CONTAINER_NAME="afo_llm_nightly_$$"
cleanup() { docker stop "${CONTAINER_NAME}" 2>/dev/null || true; docker rm "${CONTAINER_NAME}" 2>/dev/null || true; }
trap cleanup EXIT

AITER_SHA="${1:?Usage: run_afo_llm.sh <aiter_sha> <model_key> <quantizer> <tp> <threshold> [extra_args] [aiter_index_url]}"
MODEL_KEY="${2:?model_key required}"
QUANTIZER="${3:?quantizer required}"
TP="${4:?tp required}"
THRESHOLD="${5:?threshold required}"
EXTRA_ARGS="${6:-}"
AITER_INDEX_URL="${7:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve_aiter_version.sh"

VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE:-rocm/vllm-dev:nightly}"
AFO_LLM_BRANCH="${AFO_LLM_BRANCH:-main}"
AFO_LLM_REPO="${AFO_LLM_REPO:-ROCm/AFO-LLM}"
SHORT_SHA="${AITER_SHA:0:7}"
WORKDIR="${GITHUB_WORKSPACE:-.}"
IMAGE_TAG="rocm/afo-llm-ci:${SHORT_SHA}"

echo "=== AFO-LLM Accuracy Test ==="
echo "AITER SHA:   ${AITER_SHA}"
echo "Model key:   ${MODEL_KEY}"
echo "Quantizer:   ${QUANTIZER}"
echo "TP:          ${TP}"
echo "Threshold:   ${THRESHOLD}"
echo "Base image:  ${VLLM_BASE_IMAGE}"
echo ""

# ── Clone AFO-LLM ──
AFO_DIR="${WORKDIR}/afo-llm-checkout"
rm -rf "${AFO_DIR}"
CLONE_URL="https://github.com/${AFO_LLM_REPO}.git"
if [ -n "${GITHUB_TOKEN:-}" ]; then
  CLONE_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${AFO_LLM_REPO}.git"
fi
git clone --depth 1 -b "${AFO_LLM_BRANCH}" "${CLONE_URL}" "${AFO_DIR}"

# ── Build Docker image ──
cat > /tmp/Dockerfile.afo-llm <<EOF
FROM ${VLLM_BASE_IMAGE}
RUN pip uninstall -y aiter amd-aiter || true
RUN pip config set global.default-timeout 60 && pip config set global.retries 10
RUN pip install --upgrade "pybind11>=3.0.1"

RUN rm -rf /aiter && git clone https://github.com/ROCm/aiter.git /aiter && \
    cd /aiter && git checkout ${AITER_SHA} && \
    git submodule sync && git submodule update --init --recursive && \
    pip install -e .

RUN pip install lm_eval pyyaml psutil awscli 2>/dev/null || true
RUN echo "=== AITER version ===" && pip show amd-aiter || true
ENTRYPOINT [""]
EOF

docker build --network=host --no-cache \
  -t "${IMAGE_TAG}" -f /tmp/Dockerfile.afo-llm .

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
if [ -n "${AFO_LLM_ENV_VARS:-}" ]; then
  echo "${AFO_LLM_ENV_VARS}" > /tmp/afo_env_${$}.txt
  ENV_FILE_FLAG="--env-file /tmp/afo_env_${$}.txt"
fi

# ── Start container ──
docker ps -aq -f name="${CONTAINER_NAME}" | xargs -r docker stop | xargs -r docker rm || true

docker run -dt --device=/dev/kfd ${DEVICE_FLAG} \
  ${MODEL_MOUNT} ${ENV_FILE_FLAG} \
  --ipc=host --group-add video \
  --shm-size=16G --privileged \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e VLLM_ROCM_USE_AITER=1 \
  -v "${AFO_DIR}:/workspace/AFO-LLM" \
  -w /workspace/AFO-LLM \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_TAG}"

# ── Check versions ──
docker exec "${CONTAINER_NAME}" bash -lc "pip show amd-aiter vllm lm_eval 2>/dev/null || true"

# ── Build quantizer args ──
QUANT_ARGS=""
if [ "${QUANTIZER}" != "DEFAULT" ] && [ -n "${QUANTIZER}" ]; then
  QUANT_ARGS="--quantizer ${QUANTIZER}"
fi

# ── Run AFO-LLM accuracy test ──
MODEL_SAFE=$(echo "${MODEL_KEY}" | sed 's/\//_/g')
LOG_FILE="${WORKDIR}/afo_llm_nightly_${MODEL_SAFE}.log"

echo ""
echo "=== Running AFO-LLM accuracy test ==="
docker exec "${CONTAINER_NAME}" bash -lc \
  "python3 AFO_LLM.py vllm \
    --config configs/gsm8k.yaml \
    --model ${MODEL_KEY} \
    --nas-prefix /models \
    ${QUANT_ARGS} \
    ${EXTRA_ARGS}" 2>&1 | tee "${LOG_FILE}"

# ── Extract results ──
# AFO-LLM writes results JSON to its working directory
result_file=$(docker exec "${CONTAINER_NAME}" bash -lc \
  "ls -1t /workspace/AFO-LLM/*.json 2>/dev/null | head -1" || true)

if [ -z "${result_file}" ]; then
  echo "ERROR: No results JSON found"
  exit 2
fi

# Copy result file out of container
docker cp "${CONTAINER_NAME}:${result_file}" /tmp/afo_result_${$}.json

score=$(jq -r '.results.gsm8k["exact_match,flexible-extract"] // .results.gsm8k["exact_match,strict-match"] // empty' /tmp/afo_result_${$}.json 2>/dev/null || true)

if [ -z "${score}" ]; then
  # AFO-LLM may nest results differently — try alternate paths
  score=$(jq -r '.. | .["exact_match,flexible-extract"]? // empty' /tmp/afo_result_${$}.json 2>/dev/null | head -1 || true)
fi

if [ -z "${score}" ]; then
  echo "WARNING: Could not extract accuracy score from results JSON"
  echo "Results file content:"
  cat /tmp/afo_result_${$}.json
  exit 2
fi

echo ""
echo "=== Result ==="
echo "GSM8K score: ${score} (threshold: ${THRESHOLD})"

if [ "${THRESHOLD}" = "0" ]; then
  echo "BASELINE RUN: threshold=0, recording score only"
  exit 0
fi

failed=$(awk -v val="$score" -v thr="${THRESHOLD}" 'BEGIN {print (val < thr) ? 1 : 0}')
if [ "$failed" -eq 1 ]; then
  echo "FAILED: ${score} < ${THRESHOLD}"
  exit 1
fi
echo "PASSED"
