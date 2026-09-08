---
name: opus-module-build-optimization
description: Module-level JIT build-wall optimization for opus-based aiter modules. Use when an aiter JIT module's first-call build wall is a user-visible bottleneck or when adding a new module.
argument-hint: [module name or topic]
---

# Opus Module Build-Time Optimization

Techniques for reducing the JIT build wall of an aiter module bundling N opus kernel TUs + a dispatcher + a pybind TU. Patterns developed while taking `module_deepgemm_opus` from **48.4s to 11.1s** (-77%) end-to-end rebuild.

This skill is the **module-level** companion to `opus-kernel-best-practice` (which covers single-kernel codegen). Use this one when the bottleneck is host-pass header weight, TU count + parallelism, or dispatcher / pybind TU cost. Use the kernel-author skill instead when a single TU's device-pass MCG dominates.

## 0. Diagnose Before Optimizing (Most Important)

**Always run these probes first.** Speculation on what saves how much is unreliable -- module bottlenecks are non-obvious and often counter-intuitive (we expected `-mllvm` flags to matter, they didn't; we expected splitk to be slower, it was actually a hipgraph trap).

Set these once before running any snippet below:

```bash
export AITER_ROOT=$(pwd)              # your aiter checkout
export MODULE=module_<name>           # e.g. module_deepgemm_opus
export BUILD_DIR=$AITER_ROOT/aiter/jit/build/$MODULE/build
```

### End-to-end build wall (3-trial)

```bash
for i in 1 2 3; do
    rm -rf $AITER_ROOT/aiter/jit/build/$MODULE \
           $AITER_ROOT/aiter/jit/$MODULE.so
    AITER_REBUILD=1 AITER_LOG_LEVEL=info python3 -c \
        "import aiter.ops.<module_python_path>" 2>&1 | grep "finish build"
done
```

### Per-TU wall ranking (find the critical path)

```bash
cd $BUILD_DIR
for tu in *.cuda.o; do
    ninja -t commands $tu > /tmp/cmd.sh; chmod +x /tmp/cmd.sh
    rm -f $tu
    t0=$(date +%s.%N); /tmp/cmd.sh > /dev/null 2>&1; t1=$(date +%s.%N)
    printf "%5ss  %s\n" "$(awk -v t0=$t0 -v t1=$t1 'BEGIN{printf "%.2f", t1-t0}')" "$tu"
done | sort -rn | head -10
```

Top 1-2 TUs ARE the build wall under `MAX_JOBS >= N`. Optimize those; don't touch the long tail.

### Per-pass timing (host vs device)

```bash
export TU=<one_slow_tu>.cuda.o
cd $BUILD_DIR
ninja -t commands $TU > /tmp/cmd.sh
sed -i 's| -c | -ftime-report -c |' /tmp/cmd.sh; chmod +x /tmp/cmd.sh
rm -f $TU; /tmp/cmd.sh 2>&1 | tee /tmp/tr.log
```

The output has one block per pass. Read the `Total` row of each:
- **Host-pass dominant** (Front end > 5s) → header weight or template explosion → this skill (rounds 1-6)
- **Device-pass dominant** (Machine code generation > 70% of total) → kernel-internal codegen → `opus-kernel-best-practice`

### Preprocessed input size

```bash
cd $BUILD_DIR
ninja -t commands $TU > /tmp/cmd.sh
sed -E 's| -c | --cuda-host-only -E |; s| -o [^ ]+\.cuda\.o||' \
    /tmp/cmd.sh > /tmp/cmd_pp.sh; chmod +x /tmp/cmd_pp.sh
echo "host-pass lines: $(/tmp/cmd_pp.sh 2>/dev/null | wc -l)"
/tmp/cmd_pp.sh 2>/dev/null | \
    awk '/^# [0-9]+ "[^"]+"/{f=$3; gsub(/"/,"",f); next} {n[f]++}
         END {for (k in n) printf "%8d %s\n", n[k], k}' \
    | sort -rn | head -20
```

Calibration: torch-free dispatcher TU should be <150K lines. >400K means torch leaked in via some include chain.

## 1. Host/Device Pass Split in Launcher Headers

hipcc compiles every `.cu` twice (host + device). Without a guard, every per-instance launcher TU's host pass parses heavy headers AND its device pass codegens the kernel, serially in one `hipcc` invocation.

**Fix**: guard the launcher body and heavy host includes so the device pass takes the lean path. Kernel template forward declaration MUST stay visible on both passes.

```cpp
// SLOW: device pass parses torch + ATen for every TU
#include "aiter_tensor.h"
template<typename T> __global__ void my_kernel(my_kargs_t k);
template<typename T> void my_launcher(aiter_tensor_t& A, ...) { ... }

// FAST: device pass parses ~10K lines instead of ~70K
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#endif
template<typename T> __global__ void my_kernel(my_kargs_t k);   // both passes
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template<typename T> void my_launcher(aiter_tensor_t& A, ...) { ... }
#endif
```

**When NOT to apply**: single-TU kernels with no per-instance expansion -- guarding the launcher body alone is enough.

## 2. Fused Host TU + Per-File `-D__HIPCC_RTC__`

If your module has N>=5 codegen-emitted per-instance TUs, every one of them re-parses the host headers. N × parse cost adds up.

**Fix**: collapse all launchers into one `all_instances_host.cu`; emit per-instance `*.device.cu` files that carry only the `template __global__` instantiation.

```cpp
// instances/all_instances_host.cu (codegen-emitted)
#define OPUS_FUSED_HOST_TU
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>

#include "impl/kid_a.cuh"        // each .cuh detects OPUS_FUSED_HOST_TU
#include "impl/kid_b.cuh"        // and pulls only the traits header

#ifndef __HIP_DEVICE_COMPILE__
template void launcher_a<bf16_t>(...);   // explicit instantiations
template void launcher_b<fp32_t>(...);
#endif
```

Per-kid `.cuh` switches header inclusion based on the macro:

```cpp
#ifdef OPUS_FUSED_HOST_TU
#include "<traits>.cuh"                  // type defs only -- no functions
template<typename T> __global__ void kernel(my_kargs_t k);
#else
#include "<full_pipeline>.cuh"           // pipeline body for device codegen
#endif
```

Add per-source `-D__HIPCC_RTC__` for `*.device.cu` so their host pass takes the lean RTC branch:

```json
"module_<name>": {
    "flags_extra_hip_per_source": {
        "*.device.cu": ["-D__HIPCC_RTC__"]
    }
}
```

**Trap**: the fused TU MUST NOT pull in two pipeline headers that both define same-named `inline __device__` helpers -- ODR violation. Use the traits header only.

**When NOT to apply**: fewer than ~5 per-instance TUs; the codegen complexity isn't worth it.

## 3. Torch Removal in Launchers

Even after sections 1-2, fused host TU may still parse `<torch/all.h>` + ATen because the launcher takes `torch::Tensor`.

**Fix**: switch to a torch-free POD view. The canonical type in aiter is `aiter_tensor_t` (defined in `csrc/include/aiter_tensor.h`).

| Before | After |
|---|---|
| `torch::Tensor& X` | `aiter_tensor_t& X` |
| `TORCH_CHECK(...)` | `AITER_CHECK(...)` |
| `at::ScalarType::BFloat16` | `AITER_DTYPE_bf16` |
| `at::cuda::getCurrentCUDAStream()` | `aiter::getCurrentHIPStream()` |
| `return Y;` | `void` (in-place on Y) |

Python wrapper takes `@compile_ops(..., develop=True)` to auto-convert `torch.Tensor` args to `aiter_tensor_t` and inject the current stream. Keep the `return Y` contract by writing `raw(...); return Y` since C++ is `void` now.

Saves ~75% of the fused host TU's preprocessed input (e.g. 440K → 110K lines). Pattern mirrors PR #2932.

## 4. Dispatcher / Pybind TU Torch Removal

After section 3, the dispatcher or pybind TU may still be heavy because some intermediate header still pulls torch.

**Fix**: trace the include chain that opens `<torch/...>` first:

```bash
grep -n '^# 1 "[^"]*\(torch\|ATen\)' /tmp/dispatcher_pp.txt | head
# Walk back from the first torch open to find the nearest of YOUR
# headers above it -- that's the include leaking torch.
```

Replace with the lightest header that gives the same primitive types (`bf16_t`, `fp32_t`). Single `#include` swap can save several seconds.

## 5. Flat-Array Lookup Tables

If the dispatcher uses `std::unordered_map<key, std::function, ...>` for runtime kernel lookup, the heavyweight templates alone add ~1s of frontend / template instantiation per dispatcher TU.

```cpp
// SLOW: heap-allocating std::function values + hashtable templates
using Map = std::unordered_map<Tuple, std::function<...>, TupleHash>;
static const auto lookup = [] { return Map{GENERATE_LOOKUP_MACRO(...)}; }();

// FAST: zero template overhead, no first-call heap alloc
using KernelPtr = void (*)(aiter_tensor_t&, aiter_tensor_t&, ...);
struct Entry { Shape key; KernelPtr func; };
static constexpr Entry kLookup[] = {
    GENERATE_LOOKUP_MACRO(CTYPE)        // codegen emits sorted by key
};
auto it = std::lower_bound(kLookup, kLookup + N, needle, entry_less);
if (it != end && entry_eq(*it, needle)) return it->func;
```

The codegen emits entries pre-sorted by key. Function-pointer typedef replaces `std::function`.

**When NOT to apply**: dispatcher TU already <2s; the gain is bounded by `unordered_map` + `std::function` template work.

## 6. Dedicated TU for Shared Sub-Kernels

If every per-instance device.cu redundantly emits instantiations of a shared sub-kernel (e.g. a reduce kernel with N output-dtype specialisations), each TU pays the full RA + ISA emit cost; the linker dedupes the resulting weak symbols later.

**Fix**: emit one `instances/shared_kernel.device.cu` carrying all instantiations of the shared sub-kernel; remove them from each per-instance device.cu. Forward declarations in the fused host TU still resolve at link time.

Per-instance TU walls drop; CPU footprint shrinks proportionally to N. End-to-end wall doesn't always move (the slowest TU may not be the shared one), but helps under constrained `MAX_JOBS`.

**When NOT to apply**: fewer than ~5 instances of the shared sub-kernel.

## 7. Workspace Cache (Optional)

**Applies only if** section 3 left the launcher allocating a per-call workspace via `hipMallocAsync(stream)` / `hipFreeAsync(stream)` to stay torch-free. Caller-provided workspace, `torch::empty` (graph-aware caching allocator), or no workspace -- skip this section.

Not strictly a build-time optimization; it's a runtime correctness fix that the torch-removal in section 3 makes necessary. Bundle it with the build refactor or your tuner will misinterpret graph-mode allocator overhead as kernel-perf differences.

**Why**: hipgraph capture on current ROCm releases does NOT record the stream-ordered allocator as a no-op. Every replay re-runs alloc + free host-side; an aiter-style tuner running in graph mode then sees the allocator overhead instead of real kernel work.

**Suggested pattern**: replace per-call alloc with a `thread_local` cache that grows on demand. Inside capture the grow path is forbidden; `hipStreamIsCapturing` gates it.

```cpp
// SLOW: every replay re-runs alloc + free
HIP_CALL(hipMallocAsync(&ws, ws_bytes, stream));
launch_main_kernel<<<..., stream>>>(ws);
HIP_CALL(hipFreeAsync(ws, stream));

// FAST: one-time alloc, reused across replays
static thread_local void*  ws_cached_ptr   = nullptr;
static thread_local size_t ws_cached_bytes = 0;
if (ws_cached_ptr == nullptr || ws_bytes > ws_cached_bytes) {
    hipStreamCaptureStatus s = hipStreamCaptureStatusNone;
    HIP_CALL(hipStreamIsCapturing(stream, &s));
    AITER_CHECK(s == hipStreamCaptureStatusNone,
        "workspace cache miss inside HIP graph capture. "
        "Warm the cache once eagerly first.");

    if (ws_cached_ptr) {
        HIP_CALL(hipDeviceSynchronize());
        HIP_CALL(hipFree(ws_cached_ptr));
    }
    constexpr size_t kAlign = 4ull * 1024 * 1024;
    size_t grow = ((ws_bytes + kAlign - 1) / kAlign) * kAlign;
    HIP_CALL(hipMalloc(&ws_cached_ptr, grow));
    ws_cached_bytes = grow;
}
launch_main_kernel<<<..., stream>>>(ws_cached_ptr);
// no free -- cache held until thread exit
```

Callers that capture must warm the cache once eagerly (with the largest workspace they will use) BEFORE entering capture. **Alternative**: revert to `torch::empty(...)` if the module can re-accept the torch parse cost from section 3.

**Verifying the fix**: time the launcher in eager and graph modes (`torch.cuda.Event` + `torch.cuda.CUDAGraph`); if `graph >> eager` after applying section 3, this section's cache is needed.

## Anti-Patterns

- **Don't optimize without measurement.** Always run section 0 first.
- **Don't add `-mllvm` flags blindly.** Module-private overrides rarely help and can regress kernel perf. Override the aiter-global flags only after measuring both build wall AND kernel perf vs the default.
- **Don't put module-specific findings in shared headers.** Comments in `csrc/include/opus/*.hpp` (used by every opus kernel) should explain WHY a fix is there but NOT the specific kernel name or perf number that motivated it. Module-specific findings belong in the module's README.

## Summary Table

| Technique | Typical savings | Where applied |
|---|---|---|
| **Host/device pass split** (`__HIP_DEVICE_COMPILE__` + `__HIPCC_RTC__` guard) | ~50% per-TU | Every codegen-emitted launcher .cuh |
| **Fused host TU + per-source `-D__HIPCC_RTC__`** | -10s on N=38 | Modules with N>=5 per-instance TUs |
| **Torch removal in launchers** (`torch::Tensor` -> `aiter_tensor_t`) | -3s wall | Codegen launcher signatures + Python wrapper |
| **Dispatcher / pybind torch removal** (find leaking `#include`) | -5s wall | One `#include` swap, big payoff |
| **Flat-array lookup** (`unordered_map<std::function>` -> `lower_bound`) | -1s dispatcher TU FE | Dispatcher with tuned-CSV runtime lookup |
| **Dedicated TU for shared sub-kernels** | -8% per TU + CPU saving | When shared sub-kernel has >=5 instances |
| **Workspace cache** (optional) | Tuner correctness, no build-wall change | Only when section 3 leaves a per-call `hipMallocAsync` |

| Verified ineffective on `module_deepgemm_opus`, current ROCm | |
|---|---|
| `-mllvm -amdgpu-early-inline-all=false` | <1% wall delta |
| `-mllvm -amdgpu-function-calls=true` | <1% wall delta |
| `-mllvm --amdgpu-mfma-vgpr-form=false` | <1% wall delta |
| `-mllvm -greedy-regalloc-eviction-max-iterations=N` | flag rejected by ROCm 7.2.x LLVM |
