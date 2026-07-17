#!/bin/bash
# Apply local CK fix patches on top of the pinned composable_kernel submodule.
#
# This lets us test CK changes through aiter CI WITHOUT pushing anything to
# ROCm/composable_kernel: the submodule stays pinned to a valid `develop`
# commit (so the check-ck dependency gate still passes and the SHA is
# fetchable), and the diffs live as *.patch files in this repo under
# ck_patches/.
#
# Patches are generated with .github/scripts/gen_ck_patch.sh.
# Safe to run repeatedly: already-applied patches are skipped.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
CK_DIR="${REPO_ROOT}/3rdparty/composable_kernel"
PATCH_DIR="${REPO_ROOT}/ck_patches"

if [ ! -d "${PATCH_DIR}" ]; then
  echo "No ck_patches/ directory; nothing to apply."
  exit 0
fi

shopt -s nullglob
patches=("${PATCH_DIR}"/*.patch)
if [ "${#patches[@]}" -eq 0 ]; then
  echo "No *.patch files in ck_patches/; nothing to apply."
  exit 0
fi

echo "CK commit before patching: $(git -C "${CK_DIR}" rev-parse HEAD)"

for patch in "${patches[@]}"; do
  echo "=== Applying $(basename "${patch}") ==="
  if git -C "${CK_DIR}" apply --reverse --check "${patch}" 2>/dev/null; then
    echo "Already applied; skipping."
    continue
  fi
  if ! git -C "${CK_DIR}" apply --check "${patch}" 2>/dev/null; then
    echo "::error::Patch $(basename "${patch}") does not apply cleanly onto CK $(git -C "${CK_DIR}" rev-parse --short HEAD)."
    echo "The pinned CK commit may have moved; regenerate the patch with .github/scripts/gen_ck_patch.sh."
    exit 1
  fi
  git -C "${CK_DIR}" apply "${patch}"
  echo "Applied $(basename "${patch}")."
done

echo "All CK patches applied."
