#!/usr/bin/env bash
# =============================================================================
# run_vllm_test.sh — Run a vLLM pytest test inside Docker
#
# Usage:
#   run_vllm_test.sh <aiter_sha> <test_cmd> [aiter_index_url]
#
# Required env:
#   HF_TOKEN — HuggingFace token
# =============================================================================
set -euo pipefail

CONTAINER_NAME="vllm_test_$$"
cleanup() { docker stop "${CONTAINER_NAME}" 2>/dev/null || true; docker rm "${CONTAINER_NAME}" 2>/dev/null || true; }
trap cleanup EXIT

AITER_SHA="${1:?Usage: run_vllm_test.sh <aiter_sha> <test_cmd> [aiter_index_url]}"
TEST_CMD="${2:?test_cmd required}"
AITER_INDEX_URL="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve_aiter_version.sh"

VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE:-rocm/vllm-dev:nightly}"
SHORT_SHA="${AITER_SHA:0:7}"
IMAGE_TAG="rocm/vllm-aiter-ci:test-${SHORT_SHA}"

echo "=== vLLM AITER Test ==="
echo "AITER SHA: ${AITER_SHA}"
echo "Test cmd:  ${TEST_CMD}"
echo ""

# ── Build image ──
cat > /tmp/Dockerfile.vllm-test <<EOF
FROM ${VLLM_BASE_IMAGE}
RUN pip uninstall -y aiter amd-aiter || true
RUN pip config set global.default-timeout 60 && pip config set global.retries 10
RUN pip install --upgrade "pybind11>=3.0.1"
EOF

if [ -n "${AITER_INDEX_URL}" ]; then
  echo "RUN pip install --extra-index-url \"${AITER_INDEX_URL}\" \"${AITER_PKG}\"" >> /tmp/Dockerfile.vllm-test
else
  cat >> /tmp/Dockerfile.vllm-test <<EOF
RUN git clone https://github.com/ROCm/aiter.git /aiter && \
    cd /aiter && git checkout ${AITER_SHA} && \
    git submodule sync && git submodule update --init --recursive && \
    pip install -e .
EOF
fi

cat >> /tmp/Dockerfile.vllm-test <<'EOF'
RUN echo "=== AITER version ===" && pip show amd-aiter || true
EOF

docker build --network=host --no-cache \
  -t "${IMAGE_TAG}" -f /tmp/Dockerfile.vllm-test .

# ── GPU devices ──
if [ -f "/etc/podinfo/gha-render-devices" ]; then
  DEVICE_FLAG=$(cat /etc/podinfo/gha-render-devices)
else
  DEVICE_FLAG="--device /dev/dri"
fi

# ── Start container ──
docker run -dt --device=/dev/kfd ${DEVICE_FLAG} --group-add video \
  --ulimit core=0:0 --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE \
  --network=host --security-opt seccomp=unconfined --shm-size=16G \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e VLLM_ROCM_USE_AITER=1 \
  -w /app/vllm \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_TAG}"

# ── Run test ──
echo ""
echo "=== Running: ${TEST_CMD} ==="
docker exec "${CONTAINER_NAME}" bash -lc "${TEST_CMD}"

echo ""
echo "=== Test PASSED ==="
