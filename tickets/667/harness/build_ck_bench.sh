#!/usr/bin/env bash
# Build the CK warp-decode benchmark binary from commit 62e30c9098.
#
# Compiles bench_warp_decode.cpp DIRECTLY with amdclang++ — no CMake.
# CK-Tile is header-only, so the whole build is one compile+link invocation.
# This avoids pulling in tile_engine/gemm_universal and its Python instance
# builders that fail on gfx950.
#
# Usage:
#   bash build_ck_bench.sh [gfx942|gfx950|gfx942,gfx950]
#   Default arch: detected from rocminfo, falls back to gfx942;gfx950.
#
# Output binary:
#   /workspaces/rocm-libraries-wdec/bench_ck_tile_warp_decode
#
# The script sets CK_BENCH to the binary path on success.

set -euo pipefail

CK_COMMIT="62e30c9098"
ROCM_LIBS_ORIG="/workspaces/rocm-libraries"
WORKTREE_DIR="/workspaces/rocm-libraries-wdec"
CK_SRC="${WORKTREE_DIR}/projects/composablekernel"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use our minimal bench (avoids persistent_jtile_kernels_hip.hpp missing in this commit).
BENCH_SRC="${SCRIPT_DIR}/ck_bench_warp_decode.cpp"
BENCH_BIN="${WORKTREE_DIR}/bench_ck_warp_decode"
AMDCLANG="/opt/rocm/bin/amdclang++"

# ── 1. GPU arch ──────────────────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
    RAW_ARCH="${1}"
else
    RAW_ARCH="$(rocminfo 2>/dev/null | awk -F'Name:' '/Name:/{gsub(/[[:space:]]/,"",$2); if($2 ~ /^gfx[0-9]/){print $2; exit}}')"
    if [ -z "${RAW_ARCH}" ]; then
        RAW_ARCH="gfx942"
        echo "Warning: could not detect arch via rocminfo, defaulting to ${RAW_ARCH}"
    fi
fi
# amdclang++ accepts comma-separated arches in --offload-arch
OFFLOAD_ARCH="${RAW_ARCH//;/,}"
echo "Target arch: ${OFFLOAD_ARCH}"

# ── 2. Worktree ───────────────────────────────────────────────────────────────
if [ ! -d "${WORKTREE_DIR}/.git" ] && [ ! -f "${WORKTREE_DIR}/.git" ]; then
    echo "Creating worktree at ${WORKTREE_DIR} pinned to ${CK_COMMIT} ..."
    git -C "${ROCM_LIBS_ORIG}" worktree add "${WORKTREE_DIR}" "${CK_COMMIT}"
else
    echo "Worktree already exists at ${WORKTREE_DIR} — skipping."
fi

# ── 3. Direct compile (no CMake) ─────────────────────────────────────────────
echo "Compiling ${BENCH_SRC} ..."
echo "  → ${BENCH_BIN}"

"${AMDCLANG}" \
    -x hip \
    -std=c++20 \
    -O3 \
    --offload-arch="${OFFLOAD_ARCH}" \
    -DCK_TILE_USE_OCP_FP8 \
    -I "${CK_SRC}/include" \
    -Wno-global-constructors \
    -Wno-undef \
    -Wno-float-equal \
    -Wno-unused-result \
    -o "${BENCH_BIN}" \
    "${BENCH_SRC}"

# ── 4. Smoke-test ─────────────────────────────────────────────────────────────
echo ""
echo "Build successful: ${BENCH_BIN}"
echo ""
echo "Smoke-test (qwen3next, B=1, 3 iters):"
CK_WD_SHAPES="qwen3next" \
CK_WD_BATCHES="1" \
CK_WD_ITERS="3" \
CK_WD_COLD="1" \
    "${BENCH_BIN}"

echo ""
echo "CK_BENCH=${BENCH_BIN}"
export CK_BENCH="${BENCH_BIN}"
