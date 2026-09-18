#!/usr/bin/env bash
# Build and run the csrc/cpp_itfs/ gtest suites.
#
# These C++ entry points are used by non-PyTorch callers and are not covered by
# the Python test suite (which reaches the kernels through its own ctypes path),
# so they need their own check. Run from the repository root:
#
#   ./csrc/cpp_itfs/run_cpp_tests.sh
#
# Requires: hipcc, gtest, fmt, openssl, and — because the kernels are JIT-built
# from source on first call — python3 with jinja2. The first run pays a one-off
# compile cost of roughly 20s per kernel configuration; results are cached under
# ${AITER_ROOT_DIR:-$HOME}/.aiter/build/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$(mktemp -d)}"
CXXSTD="${CXXSTD:-c++20}"

cd "$REPO_ROOT"

INCLUDES=(
  -I"$REPO_ROOT/csrc/include"
  -I"$REPO_ROOT/csrc/cpp_itfs"
  -I"$REPO_ROOT/csrc"
)

# suite name -> directory under csrc/cpp_itfs/
SUITES=(
  "pa:pa"
)

failed=0
for entry in "${SUITES[@]}"; do
  name="${entry%%:*}"
  dir="csrc/cpp_itfs/${entry##*:}"

  echo "=============================================================="
  echo "== ${name}: building"
  echo "=============================================================="

  hipcc "$dir/${name}_ragged.cpp" \
    -o "$BUILD_DIR/lib${name}.so" \
    -fPIC -shared -std="$CXXSTD" -O2 \
    "${INCLUDES[@]}" -I"$dir" \
    -lfmt -lcrypto -ldl

  hipcc "$BUILD_DIR/lib${name}.so" "$dir/${name}_ragged_test.cpp" \
    -o "$BUILD_DIR/${name}_test" \
    -std="$CXXSTD" -O2 \
    "${INCLUDES[@]}" -I"$dir" \
    -lgtest -lgtest_main -lpthread -lfmt -lcrypto

  echo "== ${name}: running"
  # The JIT shells out to `python3 -m csrc.cpp_itfs...`, so it must run from the
  # repository root.
  if ! LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" "$BUILD_DIR/${name}_test"; then
    echo "== ${name}: FAILED"
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "C++ interface tests FAILED"
  exit 1
fi
echo "C++ interface tests passed"
