#!/usr/bin/env bash
# =============================================================================
# run_sglang.sh — Run a SGLang test on AITER runners
#
# Usage:
#   run_sglang.sh <aiter_sha> <test_cmd> [wheel_url]
#
# Required env:
#   HF_TOKEN — HuggingFace token
#
# Optional env:
#   SGL_BRANCH          — SGLang branch (default: v0.5.10)
#   GPU_ARCH            — GPU arch (default: gfx942)
#   SGLANG_CI_HOSTNAME  — hostname override for SGLang CI scripts
# =============================================================================
set -euo pipefail

cleanup() { docker rm -f ci_sglang 2>/dev/null || true; }
trap cleanup EXIT

AITER_SHA="${1:?Usage: run_sglang.sh <aiter_sha> <test_cmd> [wheel_url]}"
TEST_CMD="${2:?test_cmd required}"
WHEEL_URL="${3:-}"

SGL_BRANCH="${SGL_BRANCH:-v0.5.10}"
GPU_ARCH="${GPU_ARCH:-gfx942}"
SGLANG_CI_HOSTNAME="${SGLANG_CI_HOSTNAME:-linux-mi325-gpu-1}"
SGLANG_WORKSPACE="${RUNNER_TEMP:-/tmp}/sglang-downstream"

echo "=== SGLang Integration Test ==="
echo "AITER SHA:  ${AITER_SHA}"
echo "SGL branch: ${SGL_BRANCH}"
echo "Test cmd:   ${TEST_CMD}"
echo ""

# ── Clone SGLang ──
rm -rf "${SGLANG_WORKSPACE}"
git clone --depth 1 -b "${SGL_BRANCH}" https://github.com/sgl-project/sglang.git "${SGLANG_WORKSPACE}"

# ── Patch SGLang CI scripts for aiter runners ──
cd "${SGLANG_WORKSPACE}"
python3 - <<'PY'
from pathlib import Path

def replace_once(path_str, old, new):
    path = Path(path_str)
    if not path.exists():
        print(f"Warning: {path} not found, skipping patch")
        return
    text = path.read_text()
    if old not in text:
        print(f"Warning: snippet not found in {path}: {old!r}")
        return
    path.write_text(text.replace(old, new, 1))

hostname_override = 'HOSTNAME_VALUE="${SGLANG_CI_HOSTNAME_OVERRIDE:-$(hostname)}"'
for rel in (
    "scripts/ci/amd/amd_ci_start_container.sh",
    "scripts/ci/amd/amd_ci_install_dependency.sh",
    "scripts/ci/amd/amd_ci_exec.sh",
):
    replace_once(rel, "HOSTNAME_VALUE=$(hostname)", hostname_override)

replace_once(
    "scripts/ci/amd/amd_ci_install_dependency.sh",
    "install_with_retry docker exec -w /human-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e .",
    "install_with_retry docker exec -w /human-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache --no-build-isolation -e .",
)
PY

# ── Resolve SGLang base image ──
SGL_IMAGE=$(python3 - <<'PY'
import json, re, urllib.request

prefixes = ["v0.5.10-rocm700-mi30x-", "v0.5.10rc0-rocm700-mi30x-"]
matches = {p: [] for p in prefixes}
patterns = {p: re.compile(rf"^{re.escape(p)}\d{{8}}(?:-preview)?$") for p in prefixes}
next_url = "https://registry.hub.docker.com/v2/repositories/rocm/sgl-dev/tags?page_size=100"

while next_url:
    with urllib.request.urlopen(next_url, timeout=30) as resp:
        data = json.load(resp)
    for item in data.get("results", []):
        name = item["name"]
        for p in prefixes:
            if patterns[p].match(name):
                matches[p].append(name)
    next_url = data.get("next")

for p in prefixes:
    if matches[p]:
        print(f"rocm/sgl-dev:{sorted(matches[p], reverse=True)[0]}")
        exit(0)
raise SystemExit("No public MI30X SGLang image found")
PY
)
echo "SGLang image: ${SGL_IMAGE}"

# ── Start container ──
docker rm -f ci_sglang || true
export SGLANG_CI_HOSTNAME_OVERRIDE="${SGLANG_CI_HOSTNAME}"
GITHUB_WORKSPACE="${SGLANG_WORKSPACE}" HF_TOKEN="${HF_TOKEN}" \
  bash scripts/ci/amd/amd_ci_start_container.sh --custom-image "${SGL_IMAGE}"

# ── Setup pip ──
docker exec -u root ci_sglang bash -c "pip config set global.default-timeout 60"
docker exec -u root ci_sglang bash -c "pip config set global.retries 10"

# ── Install deps ──
bash scripts/ci/amd/amd_ci_install_dependency.sh --skip-aiter-build

# ── Install AITER under test ──
if [ -n "${WHEEL_URL}" ]; then
  docker exec ci_sglang bash -lc "
    pip uninstall -y amd-aiter aiter || true
    pip install --force-reinstall '${WHEEL_URL}'
    pip show amd-aiter
  "
else
  docker exec ci_sglang bash -lc "
    set -ex
    pip uninstall -y amd-aiter aiter || true
    rm -rf /tmp/aiter-under-test
    git clone https://github.com/ROCm/aiter.git /tmp/aiter-under-test
    cd /tmp/aiter-under-test
    git checkout '${AITER_SHA}'
    git submodule sync --recursive
    git submodule update --init --recursive
    PREBUILD_KERNELS=1 GPU_ARCHS='${GPU_ARCH}' python setup.py build_ext --inplace
    pip install -e .
    pip show amd-aiter || pip show aiter
  "
fi

# ── Run test ──
echo ""
echo "=== Running: ${TEST_CMD} ==="
bash scripts/ci/amd/amd_ci_exec.sh -w "/sglang-checkout" bash -c "${TEST_CMD}"

echo ""
echo "=== Test PASSED ==="
