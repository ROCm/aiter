#!/bin/bash
# Regenerate the local CK fix patch from your current working changes in the
# composable_kernel submodule.
#
# Usage:
#   1. Edit files under 3rdparty/composable_kernel/ as needed.
#   2. Run: ./.github/scripts/gen_ck_patch.sh [patch_name]
#   3. Commit the resulting ck_patches/<name>.patch on your aiter branch.
#
# CI (apply_ck_patches.sh) will re-apply this patch onto the pinned CK
# submodule so your changes get built and tested.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
CK_DIR="${REPO_ROOT}/3rdparty/composable_kernel"
PATCH_DIR="${REPO_ROOT}/ck_patches"
PATCH_NAME="${1:-local_ck_fixes}"
PATCH_FILE="${PATCH_DIR}/${PATCH_NAME}.patch"

mkdir -p "${PATCH_DIR}"

# Stage-intent so newly added files are included in the diff, then diff against
# the pinned submodule HEAD without permanently touching the index.
git -C "${CK_DIR}" add -A -N
git -C "${CK_DIR}" diff HEAD > "${PATCH_FILE}"
git -C "${CK_DIR}" reset -q

if [ ! -s "${PATCH_FILE}" ]; then
  echo "No changes detected in ${CK_DIR}; removed empty ${PATCH_FILE}."
  rm -f "${PATCH_FILE}"
  exit 0
fi

echo "Wrote CK patch: ${PATCH_FILE}"
echo "Base CK commit: $(git -C "${CK_DIR}" rev-parse HEAD)"
echo
echo "Files changed:"
git -C "${CK_DIR}" diff --stat HEAD
echo
echo "Now commit ${PATCH_FILE#${REPO_ROOT}/} on your aiter branch and push."
