#!/bin/bash
# Builds shader_clock (measures the real shader clock with clock64/wall_clock64).
# hipcc from the pip ROCm SDK hard-codes a libamdhip64.so path that may not exist,
# so compile to an object and link explicitly. Override ROCM_LIB if needed.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_LIB="${ROCM_LIB:-$(python3 -c 'import _rocm_sdk_devel, os; print(os.path.join(os.path.dirname(_rocm_sdk_devel.__file__), "lib"))')}"
hipcc --offload-arch=gfx1250 -O2 -c -o "$HERE/shader_clock.o" "$HERE/shader_clock.cpp"
clang_dir="$(dirname "$(readlink -f "$(command -v hipcc)")")/../lib/llvm/bin"
"$clang_dir/clang++" "$HERE/shader_clock.o" -L"$ROCM_LIB" -lamdhip64 -Wl,-rpath,"$ROCM_LIB" -o "$HERE/shader_clock"
echo "built $HERE/shader_clock"
