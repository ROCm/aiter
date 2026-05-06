#!/usr/bin/env bash
# =============================================================================
# run_sglang.sh — Run a SGLang test on AITER runners
#
# Usage:
#   run_sglang.sh <aiter_sha> <test_cmd> [aiter_index_url]
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

AITER_SHA="${1:?Usage: run_sglang.sh <aiter_sha> <test_cmd> [aiter_index_url]}"
TEST_CMD="${2:?test_cmd required}"
AITER_INDEX_URL="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SGLang Docker images use Python 3.10
export AITER_PYTHON_TAG=cp310

# Derive AITER wheel ROCm tag from the SGLang image's ROCm tag so that the
# wheel's c10/hip ABI matches the container's PyTorch ABI. Without this, e.g.
# a +rocm7.2 wheel installed in a rocm700 container fails at JIT .so load with
# `undefined symbol: c10::hip::c10_hip_check_implementation` (run 25202894144).
derive_aiter_rocm_tag() {
  case "$1" in
    *rocm700*|*rocm702*) echo rocm7.0 ;;
    *rocm710*|*rocm711*) echo rocm7.1 ;;
    *rocm720*|*rocm721*|*rocm722*) echo rocm7.2 ;;
    *) echo "" ;;
  esac
}
if [ -n "${SGL_IMAGE_OVERRIDE:-}" ]; then
  AITER_ROCM_TAG="$(derive_aiter_rocm_tag "${SGL_IMAGE_OVERRIDE}")"
  [ -n "${AITER_ROCM_TAG}" ] && export AITER_ROCM_TAG \
    && echo "Derived AITER_ROCM_TAG=${AITER_ROCM_TAG} from SGL_IMAGE_OVERRIDE"
fi

source "${SCRIPT_DIR}/resolve_aiter_version.sh"

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

# ── Detect GPU arch for image selection ──
GPU_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "gfx942")
case "${GPU_ARCH}" in
  gfx950) GPU_TAG="mi35x" ;;
  *)      GPU_TAG="mi30x" ;;
esac
echo "Detected GPU: ${GPU_ARCH} → image tag: ${GPU_TAG}"

# ── AITER install helpers ──
install_aiter_from_source() {
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
}

install_aiter_wheel() {
  docker exec ci_sglang bash -lc "
    pip uninstall -y amd-aiter aiter || true
    ${AITER_INSTALL_CMD}
    pip show amd-aiter
  "
}

start_container() {
  local image="$1"
  docker rm -f ci_sglang 2>/dev/null || true
  # Restore files consumed by previous candidate's install_dependency.
  # The install script does `mv pyproject_rocm.toml pyproject.toml` (and similar),
  # so these files are gone after the first candidate. We restore from the cp
  # backup taken before the loop — more reliable than `git checkout` on a shallow
  # clone whose working tree has been modified by hipify / build artifacts.
  local bkdir="${SGLANG_WORKSPACE}/.file-backups"
  if [ -d "${bkdir}" ]; then
    cp -f "${bkdir}/pyproject_rocm.toml"  "${SGLANG_WORKSPACE}/sgl-kernel/pyproject_rocm.toml" 2>/dev/null || true
    cp -f "${bkdir}/pyproject_other.toml" "${SGLANG_WORKSPACE}/python/pyproject_other.toml"    2>/dev/null || true
  fi
  export SGLANG_CI_HOSTNAME_OVERRIDE="${SGLANG_CI_HOSTNAME}"
  GITHUB_WORKSPACE="${SGLANG_WORKSPACE}" HF_TOKEN="${HF_TOKEN}" \
    bash scripts/ci/amd/amd_ci_start_container.sh --custom-image "${image}"
  docker exec -u root ci_sglang bash -c "git config --global --add safe.directory '*'"
  docker exec -u root ci_sglang bash -c "pip config set global.default-timeout 60"
  docker exec -u root ci_sglang bash -c "pip config set global.retries 10"
  bash scripts/ci/amd/amd_ci_install_dependency.sh --skip-aiter-build
}

# ── Resolve SGLang image ──
# If SGL_IMAGE_OVERRIDE is set (from preflight job), use it directly.
# Otherwise, resolve candidates by matching ROCm version from AITER_VERSION,
# then smoke-test each until one is ABI-compatible.
if [ -n "${SGL_IMAGE_OVERRIDE:-}" ]; then
  SGL_IMAGE="${SGL_IMAGE_OVERRIDE}"
  echo "Using pre-resolved SGLang image: ${SGL_IMAGE}"
  start_container "${SGL_IMAGE}"

  if [ -n "${AITER_INDEX_URL}" ]; then
    install_aiter_wheel || install_aiter_from_source
  else
    install_aiter_from_source
  fi

  # Verify AITER works on this runner (may differ from preflight runner).
  # Deep smoke: dlopen every aiter/jit/*.so via ctypes (no torch op registration,
  # no GPU required). Catches c10/hip ABI symbol mismatches that bare
  # `import aiter` misses, because aiter's top-level package does not load
  # the JIT shared objects until a tensor op is dispatched into them.
  # See run 25202894144 for the failure mode this guards against.
  docker exec ci_sglang python3 - <<'AITER_DEEP_SMOKE'
import ctypes, glob, os, sys
import aiter
jit_dir = os.path.join(os.path.dirname(aiter.__file__), "jit")
sos = sorted(glob.glob(os.path.join(jit_dir, "*.so")))
if not sos:
    sys.exit(f"AITER deep smoke FAILED: no .so found in {jit_dir} (wheel install broken)")
fails = []
for so in sos:
    try:
        ctypes.CDLL(so)  # RTLD_LOCAL — verify symbols without polluting global namespace
    except OSError as e:
        fails.append(f"  {os.path.basename(so)}: {e}")
if fails:
    print("AITER deep smoke FAILED — ABI mismatch (likely wheel ROCm != image ROCm):", file=sys.stderr)
    for f in fails:
        print(f, file=sys.stderr)
    sys.exit(1)
print(f"AITER deep smoke OK: {len(sos)} JIT .so files dlopen cleanly")
AITER_DEEP_SMOKE
else
# Resolve candidates by matching ROCm version from AITER_VERSION
readarray -t SGL_IMAGES < <(python3 - "${GPU_TAG}" "${AITER_VERSION:-}" <<'PY'
import json, re, sys, time, urllib.request, urllib.error

gpu_tag = sys.argv[1]
aiter_version = sys.argv[2] if len(sys.argv) > 2 else ""

rocm_versions = ["rocm700", "rocm720"]

prefixes = [f"v0.5.10.post1-{rv}-{gpu_tag}-" for rv in rocm_versions]
patterns = {p: re.compile(rf"^{re.escape(p)}\d{{8}}(?:-preview)?$") for p in prefixes}

max_retries = 3
for attempt in range(1, max_retries + 1):
    matches = {p: [] for p in prefixes}
    try:
        next_url = "https://registry.hub.docker.com/v2/repositories/rocm/sgl-dev/tags?page_size=100"
        while next_url:
            with urllib.request.urlopen(next_url, timeout=60) as resp:
                data = json.load(resp)
            for item in data.get("results", []):
                name = item["name"]
                for p in prefixes:
                    if patterns[p].match(name):
                        matches[p].append(name)
            next_url = data.get("next")

        found = False
        for p in prefixes:
            if matches[p]:
                print(f"rocm/sgl-dev:{sorted(matches[p], reverse=True)[0]}")
                found = True
        if not found:
            raise SystemExit(f"No public {gpu_tag} SGLang image found")
        break
    except (urllib.error.URLError, OSError) as e:
        if attempt < max_retries:
            wait = 15 * attempt
            print(f"Docker Hub request failed (attempt {attempt}/{max_retries}): {e}", file=sys.stderr)
            time.sleep(wait)
        else:
            raise SystemExit(f"Docker Hub unreachable after {max_retries} attempts: {e}")
PY
)

echo "Candidate SGLang images: ${SGL_IMAGES[*]}"

# Back up files that amd_ci_install_dependency.sh consumes via `mv`.
# The install script renames pyproject_rocm.toml → pyproject.toml (and similar),
# so after the first candidate these files are gone from the shared workspace.
# A simple cp backup is more reliable than `git checkout` on a shallow clone
# whose working tree has been modified by hipify / build artifacts.
_backup_dir="${SGLANG_WORKSPACE}/.file-backups"
mkdir -p "${_backup_dir}"
cp -f "${SGLANG_WORKSPACE}/sgl-kernel/pyproject_rocm.toml" "${_backup_dir}/" 2>/dev/null || true
cp -f "${SGLANG_WORKSPACE}/python/pyproject_other.toml"    "${_backup_dir}/" 2>/dev/null || true

# ── Try each candidate image: start container, install wheel, smoke-test import ──
SGL_IMAGE=""
for candidate in "${SGL_IMAGES[@]}"; do
  echo ""
  echo "=== Trying SGLang image: ${candidate} ==="
  start_container "${candidate}"

  if [ -n "${AITER_INDEX_URL}" ]; then
    if ! install_aiter_wheel; then
      echo "WARNING: Wheel install failed on ${candidate}, trying next image..."
      continue
    fi
  else
    if ! install_aiter_from_source; then
      echo "WARNING: Source build failed on ${candidate}, trying next image..."
      continue
    fi
  fi

  # Smoke-test: deep dlopen of every JIT .so to catch c10/hip ABI mismatches
  # that bare `import aiter` misses (see run_sglang.sh:147 for rationale).
  if docker exec ci_sglang python3 - <<'AITER_DEEP_SMOKE_CANDIDATE' 2>&1; then
import ctypes, glob, os, sys
import aiter
jit_dir = os.path.join(os.path.dirname(aiter.__file__), "jit")
sos = sorted(glob.glob(os.path.join(jit_dir, "*.so")))
if not sos:
    sys.exit(f"AITER deep smoke FAILED: no .so in {jit_dir}")
fails = []
for so in sos:
    try:
        ctypes.CDLL(so)
    except OSError as e:
        fails.append(f"  {os.path.basename(so)}: {e}")
if fails:
    print("ABI mismatch:", file=sys.stderr)
    for f in fails:
        print(f, file=sys.stderr)
    sys.exit(1)
print(f"AITER deep smoke OK: {len(sos)} JIT .so OK")
AITER_DEEP_SMOKE_CANDIDATE
    echo "AITER wheel is ABI-compatible with ${candidate}"
    SGL_IMAGE="${candidate}"
    break
  else
    echo "WARNING: AITER import failed on ${candidate} (ABI mismatch), trying next image..."
  fi
done

if [ -z "${SGL_IMAGE}" ]; then
  echo "ERROR: AITER wheel is not compatible with any available SGLang image"
  exit 1
fi

echo ""
echo "Selected SGLang image: ${SGL_IMAGE}"
fi

# Export selected image for preflight job to pass to downstream jobs
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "sgl_image=${SGL_IMAGE}" >> "${GITHUB_OUTPUT}"
fi

# ── Run test ──
echo ""
echo "=== Running: ${TEST_CMD} ==="
TEST_LOG="$(mktemp)"
set +e
bash scripts/ci/amd/amd_ci_exec.sh -w "/sglang-checkout" bash -c "${TEST_CMD}" 2>&1 | tee "${TEST_LOG}"
TEST_RC=${PIPESTATUS[0]}
set -e

# Empty-suite guard: SGLang's incremental migration prints "No tests found ...
# Skipping." and exits 0. Without this guard those zero-test runs were counted
# as PASS (run 25202894144 nightly-8gpu-deepseek-v32). Treat as SKIPPED via
# GitHub Actions neutral exit code (78) so dashboards distinguish from a real
# pass without flipping the job to red.
if grep -qE "No tests found .*Skipping\.?" "${TEST_LOG}"; then
  echo ""
  echo "::warning title=Empty Suite::No tests found for this suite — marking SKIPPED, not PASSED"
  echo "=== Test SKIPPED (no tests in suite) ==="
  exit 78
fi

if [ "${TEST_RC}" -ne 0 ]; then
  echo ""
  echo "=== Test FAILED (exit ${TEST_RC}) ==="
  exit "${TEST_RC}"
fi

echo ""
echo "=== Test PASSED ==="
