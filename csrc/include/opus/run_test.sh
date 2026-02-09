#!/bin/bash
# Build and run opus unit tests (CPU + GPU)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set -e

echo "========================================="
echo "Building opus CPU unit tests..."
hipcc -std=c++20 -O2 -I${SCRIPT_DIR}/../ ${SCRIPT_DIR}/test_opus.cpp -o ${SCRIPT_DIR}/test_opus
echo "Running CPU tests..."
${SCRIPT_DIR}/test_opus
echo ""

echo "========================================="
echo "Building opus GPU unit tests..."
hipcc -std=c++20 -O2 -I${SCRIPT_DIR}/../ ${SCRIPT_DIR}/test_opus_gpu.cpp -o ${SCRIPT_DIR}/test_opus_gpu
echo "Running GPU tests..."
${SCRIPT_DIR}/test_opus_gpu
