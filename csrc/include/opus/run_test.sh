#!/bin/bash
# Build and run opus unit tests
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set -e
echo "Building opus unit tests..."
hipcc -std=c++20 -O2 -I${SCRIPT_DIR}/../ ${SCRIPT_DIR}/test_opus.cpp -o ${SCRIPT_DIR}/test_opus
echo "Running..."
${SCRIPT_DIR}/test_opus
