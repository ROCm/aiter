#!/bin/bash
# Install the aiter under test, then flash-attention's Triton backend AGAINST it.
#
# THE ORDER IS THE POINT, which is why both halves are here. FA's setup.py initialises its own
# third_party/aiter submodule and pip-installs it -- over whatever aiter is already there,
# including the one this CI exists to test. FLASH_ATTENTION_USE_SYSTEM_AITER tells it not to.
#
# Working around that instead meant replacing FA's submodule with our checkout, deleting FA's
# .git so setup.py took its source-distribution branch, and then uninstalling and reinstalling
# aiter to undo the copy: four steps to undo one, written out per job, and they drifted --
# ROCm/aiter#5620 fixed two of the four places and left the others.
#
# Usage: install_flash_attention.sh <aiter-dir> <fa-dest> [editable]
#   editable   install flash-attention editable too (aiter always is)
#   FA_BRANCH, FA_REPOSITORY_URL   from the workflow env
#   BUILD_TARGET                   passed through if the caller exports it
set -euo pipefail

AITER="${1:?usage: install_flash_attention.sh <aiter-dir> <fa-dest> [editable]}"
DEST="${2:?usage: install_flash_attention.sh <aiter-dir> <fa-dest> [editable]}"
EDITABLE="${3:-}"

# THE AITER UNDER TEST, FIRST. Uninstalled before installing because the image may ship its own
# amd-aiter, and installing over one leaves which resolves up to the finder rather than to this
# script. `pip uninstall -y` on an absent package warns and exits 0, so this is safe either way.
pip uninstall -y amd-aiter
pip install --no-build-isolation -e "${AITER}"

# AND FLASH-ATTENTION AGAINST IT. --no-deps so resolving FA cannot pull a different amd-aiter
# over the one just installed.
git clone -b "${FA_BRANCH}" "${FA_REPOSITORY_URL}" "${DEST}"
cd "${DEST}"
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE \
    pip install --no-build-isolation --no-deps ${EDITABLE:+-e} .
