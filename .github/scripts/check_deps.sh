#!/bin/bash

CURRENT_COMMIT=$(git submodule status 3rdparty/composable_kernel | awk '{print $1}')

TEMP_DIR=$(mktemp -d)
git clone --filter=blob:none --no-checkout --single-branch --branch develop https://github.com/ROCm/composable_kernel.git "$TEMP_DIR" 2>/dev/null

# clone CK to tmp dir
if [ ! -d "$TEMP_DIR/.git" ]; then
  echo "?Clone composable_kernel failed"
  exit 1
fi

DEVELOP_COMMITS=$(git -C "$TEMP_DIR" rev-list develop)

# echo "CURRENT COMMIT: $CURRENT_COMMIT"
# echo "DEVELOP COMMIT: $DEVELOP_COMMITS"

if ! echo "$DEVELOP_COMMITS" | grep -q "^$CURRENT_COMMIT$"; then
  echo "?3rdparty/composable_kernel must be commits from develop branch"
  echo "CURRENT COMMIT: $CURRENT_COMMIT"
  exit 1
else
  echo "?Reference to composable_kernel is valid"
fi
