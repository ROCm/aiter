#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Build + run the torch-free batched_gemm_bf16 proof of concept.
#   1. compile.py builds libbatched_gemm_bf16.so (torch_exclude, C-ABI).
#   2. hipcc links the C++ test against it (no torch).
#   3. verify the .so has no libtorch/libc10 linkage, then run.
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOP_DIR=$(dirname "$SCRIPT_DIR")/../../

echo "######## [1/3] building torch-free libbatched_gemm_bf16.so"
cd "$SCRIPT_DIR"
python3 compile.py

# build_module copies standalone (non-python) .so artifacts into op_tests/cpp/mha.
# Pull ours next to the test so -L./ and $ORIGIN rpath find it.
SO_NAME=libbatched_gemm_bf16.so
SO_SRC="$TOP_DIR/op_tests/cpp/mha/$SO_NAME"
if [ -f "$SO_SRC" ]; then
    cp "$SO_SRC" "$SCRIPT_DIR/"
fi
if [ ! -f "$SCRIPT_DIR/$SO_NAME" ]; then
    echo "ERROR: $SO_NAME not found (looked in $SO_SRC and $SCRIPT_DIR)"
    exit 1
fi

echo "######## [2/3] verifying no torch linkage"
if ldd "$SCRIPT_DIR/$SO_NAME" | grep -qiE "libtorch|libc10"; then
    echo "WARNING: torch/c10 IS linked into $SO_NAME:"
    ldd "$SCRIPT_DIR/$SO_NAME" | grep -iE "libtorch|libc10"
else
    echo "OK: no libtorch / libc10 in $SO_NAME"
fi

echo "######## [3/3] linking + running C++ test"
# rpath: $ORIGIN finds the .so next to the exe; /opt/rocm/lib finds libamdhip64.
/opt/rocm/bin/hipcc -I"$TOP_DIR/csrc/cpp_itfs/batched_gemm_bf16" \
                    -std=c++20 -O3 \
                    -DUSE_ROCM=1 \
                    --offload-arch=native \
                    -L"$SCRIPT_DIR" -lbatched_gemm_bf16 \
                    -Wl,-rpath,'$ORIGIN':/opt/rocm/lib \
                    "$SCRIPT_DIR/test_batched_gemm_bf16.cpp" -o "$SCRIPT_DIR/test_bgemm.exe"

"$SCRIPT_DIR/test_bgemm.exe"
echo ""
echo "######## extra shapes"
"$SCRIPT_DIR/test_bgemm.exe" 4 2048 2048 2048
"$SCRIPT_DIR/test_bgemm.exe" 8 32 1024 1024
