#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Usage: bash build_hipblaslt_public_bridge.sh <hipblaslt-prefix> <output>
# The prefix must contain include/hipblaslt and lib/libhipblaslt.so.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: $0 <hipblaslt-prefix> <output>" >&2
  exit 2
fi

hipblaslt_dir=$(realpath "$1")
output=$(realpath -m "$2")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rocm_dir="${ROCM_PATH:-/opt/rocm}"
compiler="${CXX:-$rocm_dir/bin/amdclang++}"
python="${PYTHON:-python3}"
python_include=$(
  "$python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])'
)
pybind_include=$(
  "$python" -c 'import pybind11; print(pybind11.get_include())'
)

"$compiler" -std=c++17 -O2 -fPIC -shared -D__HIP_PLATFORM_AMD__=1 \
  -I"$hipblaslt_dir/include" -I"$rocm_dir/include" \
  -I"$python_include" -I"$pybind_include" \
  "$script_dir/csrc/hipblaslt_public_bridge.cc" \
  -L"$hipblaslt_dir/lib" -lhipblaslt \
  -L"$rocm_dir/lib" -lamdhip64 \
  -Wl,-rpath,"$hipblaslt_dir/lib" -Wl,-rpath,"$rocm_dir/lib" \
  -o "$output"
