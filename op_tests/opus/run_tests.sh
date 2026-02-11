#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Run all OPUS tests:
#   1. C++ host test (test_opus_basic)
#   2. Device kernel tests via opus_device_test PyTorch extension
#      (MFMA 32x32x8 fp16, vector_add)
#
# Invoke from op_tests/opus, e.g.:
#   ./run_tests.sh
# or from Docker: cd /raid0/carhuang/repo/aiter/op_tests/opus && ./run_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=python3
if ! command -v python3 &>/dev/null; then
  PYTHON=python
fi

echo "=== OPUS tests (workdir: $SCRIPT_DIR) ==="

echo ""
echo "--- C++ host test (test_opus_basic) ---"
./build.sh --test

echo ""
echo "--- OPUS device kernel tests (opus_device_test) ---"
$PYTHON device/test_opus_device.py

echo ""
echo "=== All OPUS tests finished ==="
