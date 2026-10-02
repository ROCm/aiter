#!/usr/bin/env bash
# Variables below are also consumed by the scripts sourcing this file.
# shellcheck disable=SC2034

case "${GPU_ARCHS:-}" in
    gfx942|gfx950) ;;
    *) echo "Expected GPU_ARCHS=gfx942 or gfx950" >&2; exit 1 ;;
esac
if [[ "${HSA_XNACK:-}" != 1 ]]; then
    echo "GPU ASan requires HSA_XNACK=1" >&2
    exit 1
fi

rocm_path=$(readlink -f /opt/rocm)
compiler="${rocm_path}/llvm/bin/clang++"
asan_runtime=$("$compiler" -print-file-name=libclang_rt.asan-x86_64.so)
test -f "$asan_runtime"
asan_runtime_dir=$(dirname "$asan_runtime")
asan_preload="${asan_runtime}:${rocm_path}/lib/asan/libamdhip64.so.7:${rocm_path}/lib/asan/libhsa-runtime64.so.1"
normal_library_path="${rocm_path}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
torch_library_path=$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("torch").origin).parent / "lib")')
export PATH="${rocm_path}/llvm/bin:${rocm_path}/bin:${PATH}"
# ASan intercepts dlopen, so PyTorch's caller-relative RPATH is not sufficient.
export LD_LIBRARY_PATH="${rocm_path}/lib/asan:${asan_runtime_dir}:${torch_library_path}:${rocm_path}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ASAN_SYMBOLIZER_PATH="${rocm_path}/llvm/bin/llvm-symbolizer"
export ASAN_OPTIONS=detect_leaks=0:halt_on_error=1
test -x "$ASAN_SYMBOLIZER_PATH"
