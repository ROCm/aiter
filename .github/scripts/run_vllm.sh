#!/usr/bin/env bash
# =============================================================================
# run_vllm.sh — Build vLLM+AITER image and run latency benchmark
#
# Usage:
#   run_vllm.sh <aiter_sha> <model> <tp> [kv_cache_dtype] [aiter_index_url]
#
# Required env:
#   HF_TOKEN — HuggingFace token for model access
#
# Optional env:
#   VLLM_BASE_IMAGE — default: rocm/vllm-dev:nightly
# =============================================================================
set -euo pipefail

cleanup() { docker rmi "rocm/vllm-aiter-ci:nightly-${AITER_SHA:0:7}" 2>/dev/null || true; }
trap cleanup EXIT

AITER_SHA="${1:?Usage: run_vllm.sh <aiter_sha> <model> <tp> [kv_cache_dtype] [aiter_index_url]}"
MODEL="${2:?model required}"
TP="${3:?tp required}"
KV_CACHE_DTYPE="${4:-default}"
AITER_INDEX_URL="${5:-}"

VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE:-rocm/vllm-dev:nightly}"
SHORT_SHA="${AITER_SHA:0:7}"
IMAGE_TAG="rocm/vllm-aiter-ci:nightly-${SHORT_SHA}"
CONTAINER_NAME="vllm_nightly_$$"

echo "=== vLLM Latency Benchmark ==="
echo "AITER SHA:   ${AITER_SHA}"
echo "Model:       ${MODEL}"
echo "TP:          ${TP}"
echo "KV cache:    ${KV_CACHE_DTYPE}"
echo "Base image:  ${VLLM_BASE_IMAGE}"
echo ""

# ── Build image ──
cat > /tmp/Dockerfile.vllm-nightly <<EOF
FROM ${VLLM_BASE_IMAGE}
RUN pip uninstall -y aiter amd-aiter || true
RUN pip config set global.default-timeout 60 && pip config set global.retries 10
RUN pip install --upgrade "pybind11>=3.0.1"
EOF

if [ -n "${AITER_INDEX_URL}" ]; then
  echo "RUN pip install --extra-index-url \"${AITER_INDEX_URL}\" amd-aiter" >> /tmp/Dockerfile.vllm-nightly
else
  cat >> /tmp/Dockerfile.vllm-nightly <<EOF
RUN git clone https://github.com/ROCm/aiter.git /aiter && \
    cd /aiter && git checkout ${AITER_SHA} && \
    git submodule sync && git submodule update --init --recursive && \
    pip install -e .
EOF
fi

cat >> /tmp/Dockerfile.vllm-nightly <<'EOF'
RUN echo "=== AITER version ===" && pip show amd-aiter || true
ENTRYPOINT [""]
EOF

docker build --network=host --no-cache \
  -t "${IMAGE_TAG}" -f /tmp/Dockerfile.vllm-nightly .

# ── Resolve GPU devices ──
if [ -f "/etc/podinfo/gha-render-devices" ]; then
  DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
else
  DEVICE_FLAG="--device /dev/dri"
fi

# ── Build extra args ──
EXTRA_ARGS=""
case "${MODEL}" in *DeepSeek*) EXTRA_ARGS="--block-size 1" ;; esac
if [ "${KV_CACHE_DTYPE}" = "fp8" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --kv-cache-dtype fp8"
fi

# ── Run benchmark ──
MODEL_SAFE=$(echo "${MODEL}" | sed 's/\//_/g')
LOG_FILE="vllm_nightly_${MODEL_SAFE}_tp${TP}_${KV_CACHE_DTYPE}.log"

docker run --rm --device=/dev/kfd ${DEVICE_FLAG} --group-add video \
  --ulimit core=0:0 --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE \
  --network=host --security-opt seccomp=unconfined --shm-size=16G \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e VLLM_ROCM_USE_AITER=1 \
  "${IMAGE_TAG}" \
  python -m vllm.entrypoints.cli.main bench latency \
    --model "${MODEL}" \
    --batch-size 123 --input-len 456 --output-len 78 \
    --num-iters-warmup 3 --num-iters 10 \
    -tp "${TP}" --load-format dummy ${EXTRA_ARGS} 2>&1 \
  | tee "${LOG_FILE}"

echo ""
echo "=== Results ==="
grep "Avg latency:" "${LOG_FILE}" || echo "No latency result found"

# ── Cleanup ──
docker rmi "${IMAGE_TAG}" || true
