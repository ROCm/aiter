#!/bin/bash

# Compile and run direct HIP kernel test for causal_conv1d_update

set -e

echo "=== Building Direct Conv1D Test ==="

# Set ROCm path
ROCM_PATH=${ROCM_PATH:-/opt/rocm}

# Compiler settings
HIPCC=$ROCM_PATH/bin/hipcc
CXX_FLAGS="-O3 -std=c++17"
ARCH_FLAGS="--offload-arch=gfx942"  # For MI308

# Source and output
SRC_FILE="test_direct_conv1d.cpp"
OUT_FILE="test_direct_conv1d"

# Compile
echo "Compiling $SRC_FILE..."
$HIPCC $CXX_FLAGS $ARCH_FLAGS $SRC_FILE -o $OUT_FILE

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo ""
    echo "=== Running Test ==="
    ./$OUT_FILE
else
    echo "Compilation failed!"
    exit 1
fi

