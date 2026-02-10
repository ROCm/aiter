#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Run all OPUS tests (C++ host test + MFMA PyTorch extension test).
# Invoke from op_tests/opus, e.g.:
#   ./run_tests.sh
# or from Docker: cd /raid0/carhuang/repo/aiter/op_tests/opus && ./run_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== OPUS tests (workdir: $SCRIPT_DIR) ==="

echo ""
echo "--- C++ host test (test_opus_basic) ---"
./build.sh --test

echo ""
echo "--- OPUS MFMA PyTorch extension test ---"
if command -v python3 &>/dev/null; then
  python3 mfma/test_opus_mfma.py
else
  python mfma/test_opus_mfma.py
fi

echo ""
echo "=== All OPUS tests finished ==="
