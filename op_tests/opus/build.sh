#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Build (and optionally run) OPUS C++ tests.
# test_opus_basic = standalone executable; test_opus_mfma is built as PyTorch ext (see test_opus_mfma.py).
# Can be invoked from any directory.
# Usage:
#   ./build.sh           - compile test_opus_basic
#   ./build.sh --test    - compile and run test_opus_basic
#   ./build.sh --clean   - remove built executables

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

HIPCC="${HIPCC:-hipcc}"
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -I${REPO_ROOT}/csrc/include"

TESTS=(test_opus_basic)
SOURCES=(test_opus_basic.cpp)

build() {
  if ! command -v "$HIPCC" &>/dev/null; then
    echo "Warning: $HIPCC not found. Tests require ROCm/HIP to compile."
    echo "Please install ROCm or set HIPCC to your hipcc path."
    exit 1
  fi
  echo "Using $HIPCC for compilation"
  for i in "${!TESTS[@]}"; do
    echo "Building ${TESTS[$i]}..."
    "$HIPCC" $CXXFLAGS "${SOURCES[$i]}" -o "${TESTS[$i]}"
  done
  echo "Build complete."
}

run_tests() {
  echo "======================================"
  echo "Running OPUS Unit Tests"
  echo "======================================"
  for t in "${TESTS[@]}"; do
    echo ""
    echo "Running $t..."
    ./"$t" || exit 1
  done
  echo ""
  echo "======================================"
  echo "All tests passed!"
  echo "======================================"
}

clean() {
  for t in "${TESTS[@]}"; do
    rm -f "$t"
  done
  echo "Cleaned build artifacts."
}

case "${1:-}" in
  --test)
    build
    run_tests
    ;;
  --clean)
    clean
    ;;
  --help|-h)
    echo "Usage: $0 [--test|--clean|--help]"
    echo "  (no args)  - compile only"
    echo "  --test     - compile and run tests"
    echo "  --clean    - remove executables"
    echo "  --help     - show this help"
    ;;
  "")
    build
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Use --help for usage."
    exit 1
    ;;
esac
