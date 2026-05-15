#!/usr/bin/env bash
# Apply the DeepSeek-V4-Flash optimization stack to a fresh checkout.
#
# Usage:
#   ATOM_PATH=/path/to/ATOM AITER_PATH=/path/to/aiter ./apply.sh
#
# Idempotent: dry-runs first, only applies if the patch fits cleanly.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_PATH="${ATOM_PATH:?ATOM_PATH must point to the ATOM checkout}"
AITER_PATH="${AITER_PATH:?AITER_PATH must point to the aiter checkout}"

echo "[apply] ATOM=$ATOM_PATH  AITER=$AITER_PATH"
echo "[apply] dry-run aiter patch..."
cd "$AITER_PATH"
git apply --check "$HERE/aiter_optimizations.patch" || {
  echo "[apply] aiter patch does NOT apply cleanly. Likely the new base branch has diverged."
  echo "[apply] Inspect $HERE/aiter_optimizations.patch and resolve manually per the per-step file lists in HANDOFF.md."
  exit 1
}
echo "[apply] dry-run ATOM patch..."
cd "$ATOM_PATH"
git apply --check "$HERE/atom_optimizations.patch" || {
  echo "[apply] ATOM patch does NOT apply cleanly. Likely the new base branch has diverged."
  echo "[apply] Inspect $HERE/atom_optimizations.patch and resolve manually per the per-step file lists in HANDOFF.md."
  exit 1
}

echo "[apply] applying aiter patch..."
cd "$AITER_PATH" && git apply "$HERE/aiter_optimizations.patch"
echo "[apply] applying ATOM patch..."
cd "$ATOM_PATH" && git apply "$HERE/atom_optimizations.patch"

echo "[apply] DONE. Working trees modified — review with 'git status' and 'git diff' in each repo."
echo "[apply] Next: run validate.sh to test gsm8k + perf."
