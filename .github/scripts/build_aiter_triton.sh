#!/bin/bash

set -ex

retry_cmd() {
    local max_attempts="$1"
    shift
    local attempt=1
    local rc=0

    while true; do
        if "$@"; then
            return 0
        fi
        rc=$?
        if [[ "$attempt" -ge "$max_attempts" ]]; then
            echo "Command failed after ${attempt} attempts: $*"
            return "$rc"
        fi
        local sleep_seconds=$((attempt * 20))
        echo "Attempt ${attempt}/${max_attempts} failed; retrying in ${sleep_seconds}s..."
        sleep "${sleep_seconds}"
        attempt=$((attempt + 1))
    done
}

echo
echo "==== ROCm Packages Installed ===="
dpkg -l | grep rocm || echo "No ROCm packages found."

echo
echo "==== Install dependencies and aiter ===="
git config --global --add safe.directory /workspace
pip config set global.retries 15
pip config set global.timeout 120
retry_cmd 3 pip install -r .github/requirements/triton-test.txt

# ###################################################################
# # EXPERIMENT (bmazzott): force Triton to be COMPILED FROM SOURCE. #
# # Throwaway hack -- do NOT merge. See notes below.                #
# ###################################################################
#
# CI normally injects TRITON_WHEEL_DIR=/workspace/triton_wheels (populated with a
# prebuilt wheel by the workflow). That short-circuits both install_triton.sh and
# the old BUILD_TRITON block into installing the wheel. Nuke it so nothing can
# grab a prebuilt wheel behind our back.
unset TRITON_WHEEL_DIR
export TRITON_WHEEL_DIR=""

# .github/scripts/install_triton.sh

echo
echo "##################################################################"
echo "## EXPERIMENT: BUILDING TRITON FROM SOURCE (THROWAWAY HACK)      ##"
echo "##################################################################"
pip uninstall -y triton triton-kernels pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true

# Pin triton to a known commit so the build is reproducible.
# Commit from June 24th, 2026 - [CI] Upgrade AMD CI to PyTorch 2.10 (#10625)
TRITON_COMMIT='7917e3f62f17d4329b81c29e74d0bc6b8a9963c8'
echo "[experiment] Target TRITON_COMMIT=${TRITON_COMMIT}"

# Network in CI is flaky and a full clone of triton routinely times out
# (curl 56 / "early EOF"). Do a shallow single-commit fetch and retry it.
# Run inside a subshell so a failed attempt doesn't leave us cd'd into a
# half-written dir; retry_cmd then re-runs the whole thing cleanly.
fetch_triton() {
    rm -rf triton
    mkdir -p triton
    (
        cd triton
        git init -q
        git remote add origin https://github.com/triton-lang/triton
        git fetch --depth 1 origin "$TRITON_COMMIT"
        git checkout -q FETCH_HEAD
    )
}
retry_cmd 5 fetch_triton

cd triton
echo "[experiment] triton checked out at HEAD=$(git rev-parse HEAD)"
retry_cmd 3 pip install -r python/requirements.txt
MAX_JOBS=64 pip --retries=10 --default-timeout=60 install .
cd ..
echo "[experiment] Finished building Triton from source."

# Install aiter WITHOUT letting setup.py replace our freshly-built source Triton.
# setup.py calls install_triton.sh unless AITER_USE_SYSTEM_TRITON=1 (and a triton
# is already present), so set it now that the source build is installed.
export AITER_USE_SYSTEM_TRITON=1
pip uninstall -y aiter || true
retry_cmd 3 pip install --no-build-isolation -e .

echo
echo "==== Verify triton (should be the SOURCE build) ===="
python .github/scripts/verify_triton_pin.py
echo "==== [experiment] Confirm Triton provenance ===="
python -c "import triton, os; print('[experiment] triton', triton.__version__, 'imported from', os.path.dirname(triton.__file__))"

echo
echo "==== Show installed packages ===="
pip list
