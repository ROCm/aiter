#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=.github/scripts/asan/env.sh
source .github/scripts/asan/env.sh

mode=${1:-asan}
case "$mode" in
    asan)
        export LD_PRELOAD="$asan_preload"
        export AITER_USE_ASAN=1
        log_file=asan_logs/activation.log
        ;;
    baseline)
        export LD_PRELOAD=''
        export LD_LIBRARY_PATH="$normal_library_path"
        export AITER_USE_ASAN=0
        log_file=asan_logs/activation-baseline.log
        ;;
    *) echo "Expected asan or baseline" >&2; exit 2 ;;
esac
export AITER_JIT_DIR="/workspace/${mode}_jit"
export AITER_LOG_MORE=1
export PYTORCH_NO_HIP_MEMORY_CACHING=1

# Keep the inputs identical when comparing sanitizer and ordinary builds.
timeout 25m python3 - -m 1 -n 1024 -d bf16 <<'PY' 2>&1 | tee "$log_file"
import runpy
import torch

torch.manual_seed(0)
runpy.run_path("op_tests/test_activation.py", run_name="__main__")
PY

# checkAllclose can log a numerical failure without raising an exception.
if grep -E '\[checkAllclose.*(failed!|catastrophic!)' "$log_file"; then
    echo "AITER activation correctness check failed" >&2
    exit 1
fi

if [[ "$mode" == baseline ]]; then
    echo "AITER activation baseline passed without ASan"
    exit 0
fi

module="${AITER_JIT_DIR}/asan/module_activation.so"
test -f "$module"
"${rocm_path}/llvm/bin/llvm-readelf" --dynamic "$module" | tee asan_logs/activation-linkage.log
grep -F libclang_rt.asan-x86_64.so asan_logs/activation-linkage.log
strings "$module" | grep -F "amdhsa--${GPU_ARCHS}:xnack+" \
    > asan_logs/activation-targets.log
cat asan_logs/activation-targets.log
echo "AITER activation smoke test passed with ASan and xnack+"
