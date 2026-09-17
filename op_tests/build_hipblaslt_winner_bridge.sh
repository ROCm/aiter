#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Usage: bash build_hipblaslt_winner_bridge.sh <rocm-libraries> <build> <out.so>
# Build the TensileLite preset first. ROCM_PATH must include development headers
# and unversioned libraries. GPU_ARCH defaults to gfx1250.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: $0 <rocm-libraries-source> <hipblaslt-build> <output.so>" >&2
  exit 2
fi
source_dir=$(realpath "$1")
build_dir=$(realpath "$2")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tensile_dir="$source_dir/projects/hipblaslt/tensilelite"
rocm_dir="${ROCM_PATH:-/opt/rocm}"
compiler="${CXX:-amdclang++}"
python="${PYTHON:-python3}"
python_include=$(
  "$python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])'
)
pybind_include=$(
  "$python" -c 'import pybind11; print(pybind11.get_include())'
)
"$compiler" -x hip --offload-arch="${GPU_ARCH:-gfx1250}" \
  --hip-path="$rocm_dir" -std=c++20 -O2 -shared -fPIC \
  -D__HIP_PLATFORM_AMD__=1 -DTENSILE_YAML \
  -I"$python_include" -I"$pybind_include" \
  -I"$rocm_dir/include" -I"$tensile_dir/include" \
  -I"$build_dir/tensilelite/include" -I"$tensile_dir/rocisa" \
  -I"$source_dir/shared/origami/include" -I"$build_dir/origami/include" \
  "$script_dir/csrc/hipblaslt_winner_bridge.cc" \
  -L"$build_dir/tensilelite" -ltensilelite-host \
  -L"$rocm_dir/lib" -lamdhip64 \
  -Wl,-rpath,"$build_dir/tensilelite" -Wl,-rpath,"$rocm_dir/lib" \
  -o "$3"
