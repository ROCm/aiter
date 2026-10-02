#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=.github/scripts/asan/env.sh
source .github/scripts/asan/env.sh

mkdir -p asan_logs

{
    uname -a
    dpkg-query -W rocm-core rocm-llvm hip-runtime-amd-asan hsa-rocr-asan comgr-asan
    "$compiler" --version
    printf 'GPU_ARCHS=%s\nHSA_XNACK=%s\n' "$GPU_ARCHS" "$HSA_XNACK"
    printf 'ASan runtime: %s\n' "$asan_runtime"
    printf 'amdgpu noretry: '
    cat /sys/module/amdgpu/parameters/noretry || true
    ulimit -a
    free -h
    for file in memory.max memory.current memory.events; do
        if [[ -f "/sys/fs/cgroup/${file}" ]]; then
            printf '%s: ' "$file"
            cat "/sys/fs/cgroup/${file}"
        fi
    done
} | tee asan_logs/environment.log

LD_PRELOAD="$asan_runtime" timeout 120 rocminfo > asan_logs/rocminfo.log 2>&1
"$compiler" -x hip --offload-arch="${GPU_ARCHS}:xnack+" \
    -fsanitize=address -shared-libsan -g -O1 -fno-omit-frame-pointer \
    -Werror=option-ignored \
    .github/scripts/asan/gpu_asan_probe.hip -o asan_logs/gpu_asan_probe \
    2>&1 | tee asan_logs/compile.log

ldd asan_logs/gpu_asan_probe | tee asan_logs/linked-libraries.log
for library in libamdhip64 libhsa-runtime64 libamd_comgr; do
    if ! grep -E "${library}.*=> .*/lib/asan/" asan_logs/linked-libraries.log; then
        echo "Not using the ASan ${library} runtime" >&2
        exit 1
    fi
done
grep -F libclang_rt.asan-x86_64.so asan_logs/linked-libraries.log

timeout 120 asan_logs/gpu_asan_probe valid "$GPU_ARCHS" 2>&1 | tee asan_logs/valid.log
set +e
timeout 120 asan_logs/gpu_asan_probe invalid "$GPU_ARCHS" > asan_logs/invalid.log 2>&1
status=$?
set -e
cat asan_logs/invalid.log
if [[ "$status" != 1 ]] || ! grep -Eq \
    'ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device' asan_logs/invalid.log; then
    echo "Expected a GPU ASan heap-buffer-overflow and exit 1, got exit ${status}" >&2
    exit 1
fi
grep -E 'asan_probe.*gpu_asan_probe.hip:' asan_logs/invalid.log
echo "GPU ASan positive and negative controls passed"

# ROCm ASan resolves HSA allocation functions with RTLD_NEXT. Load HIP/HSA
# globally at startup, rather than waiting for Python to dlopen them locally.
if ! LD_PRELOAD="$asan_preload" PYTORCH_NO_HIP_MEMORY_CACHING=1 timeout 120 \
    python3 -u .github/scripts/asan/pytorch_smoke.py 2>&1 | tee asan_logs/pytorch.log; then
    # Diagnose separately without converting the primary failure to a pass.
    LD_PRELOAD='' LD_LIBRARY_PATH="$normal_library_path" \
        PYTORCH_NO_HIP_MEMORY_CACHING=1 timeout 120 \
        python3 -u .github/scripts/asan/pytorch_smoke.py \
        > asan_logs/pytorch-baseline.log 2>&1 || true
    LD_PRELOAD="$asan_preload" ASAN_OPTIONS="${ASAN_OPTIONS}:allocator_may_return_null=1" \
        PYTORCH_NO_HIP_MEMORY_CACHING=1 timeout 120 \
        python3 -u .github/scripts/asan/pytorch_smoke.py \
        > asan_logs/pytorch-null-allocator.log 2>&1 || true
    cat asan_logs/pytorch-baseline.log asan_logs/pytorch-null-allocator.log
    exit 1
fi
