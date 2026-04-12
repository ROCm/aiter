---
name: opus-kernel-best-practice
description: Compile-time optimization guidance for HIP/C++ kernels using opus.hpp. Use when writing or reviewing OPUS kernels, analyzing compile time, reducing template instantiation overhead, or optimizing hipcc build performance.
argument-hint: [file or topic]
---

# OPUS Kernel Compile-Time Best Practices

Techniques for reducing HIP/C++ kernel compile time when using `opus.hpp`. These patterns were developed while optimizing a GQA flash attention kernel from **4.8s to 1.5s** (70% reduction) in device-only compilation.

## 1. Minimize Header Overhead

### Replace `<hip/hip_runtime.h>` with minimal declarations + compiler builtins

Standard `<hip/hip_runtime.h>` expands to ~190K preprocessed lines. Replace with a ~60-line `hip_minimal.h` using AMDGCN builtins:

```cpp
// Instead of:
#include <hip/hip_runtime.h>
int tid = threadIdx.x;

// Use:
int tid = __builtin_amdgcn_workitem_id_x();
int bid = __builtin_amdgcn_workgroup_id_x();
int bsz = __builtin_amdgcn_workgroup_size_x();
__builtin_amdgcn_s_barrier();  // __syncthreads()
```

### Use `-D__HIPCC_RTC__` to suppress implicit includes

Even with minimal headers, hipcc's implicit `__clang_hip_runtime_wrapper.h` pulls in `<cmath>`, `<cstdlib>`, etc. The `-D__HIPCC_RTC__` flag skips these. Provide `#define INFINITY __builtin_huge_valf()` if needed.

### Guard device code with `__HIP_DEVICE_COMPILE__`

hipcc compiles each `.hip`/`.cu` file in **two passes** (host + device). The heavy `opus.hpp` is only needed on the device side:

```cpp
#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"
__global__ void my_kernel(...) { /* uses opus */ }
#else
#include "hip_host_minimal.h"
__global__ void my_kernel(...);  // declaration only
extern "C" void run_kernel(...) { hipLaunchKernelGGL(my_kernel, ...); }
#endif
```

This avoids parsing opus.hpp on the host pass (saves ~50% of frontend time).

### Use ctypes instead of pybind11/torch extension for Python bindings

The C++ binding layer is often the biggest compile cost. Replace pybind11/torch `CUDAExtension` with `extern "C"` host launchers loaded via `ctypes.CDLL`:

| Binding | Compile time |
|---------|-------------|
| torch `CUDAExtension` | ~21s |
| pybind11 + Ninja | ~4.2s |
| ctypes (`extern "C"`) | ~0.4s |

## 2. Reduce Template Instantiation Count

### Use runtime loops instead of `static_for` where compile-time indices aren't needed

Each iteration of `static_for<N>([&](auto I){...})` creates a unique lambda instantiation. For large N, this dominates compile time. Replace with plain `for` loops when the loop body doesn't need compile-time `I`:

```cpp
// SLOW: N unique lambda instantiations
static_for<N>([&](auto I) {
    r[I.value] = load<vec>(offsets[I.value]);
});

// FAST: 1 instantiation, compiler unrolls identically
for (index_t i = 0; i < N; i++) {
    r[i] = load<vec>(offsets[i]);
}
```

**When you still need `static_for`**: If the body uses `I` as a template argument (e.g., `number<I.value>{}` for `set_slice`, `slice`, or immediate-offset `_tr_load<vec, off>`), you must keep `static_for`.

### Use runtime `flat_to_coords` instead of compile-time multi-index decomposition

`layout_to_offsets` converts a layout into a precomputed offset array using a runtime loop with `flat_to_coords`, which produces `tuple<index_t, ...>` (one type for all iterations) instead of `tuple<number<a>, number<b>, ...>` (unique type per iteration):

```cpp
// SLOW: N unique coord_to_linear instantiations (one per multi-index combination)
static_ford(issue_space_vec, [&](auto... ids) {
    offsets[u_linear(ids...)] = u(ids...);
});

// FAST: 1 coord_to_linear instantiation (all iterations use tuple<index_t, ...>)
for (index_t i = 0; i < num_issues; i++) {
    offsets[i] = u(flat_to_coords(i, make_index_seq<ndim>{}, issue_space_vec));
}
```

### Cache constexpr computations in struct members

Repeated constexpr evaluations in multiple methods trigger re-evaluation in each:

```cpp
// SLOW: y_shape_a() + reduce_tuple_mul evaluated in every operator()/step_k() overload
constexpr auto a_len = get<0>(reduce_tuple_mul(MMA::y_shape_a()));

// FAST: cached once as class member
static constexpr index_t mma_a_len = get<0>(reduce_tuple_mul(MMA::y_shape_a())).value;
```

## 3. Use LLVM Builtins for Vector Operations

### `__builtin_convertvector` for type conversion

Replaces N-element element-by-element `cast_impl` pack expansion with a single LLVM intrinsic:

```cpp
// SLOW: 64-element pack expansion
return vector_return_type<D, decltype(cast<D>(get<Is>(s)))...>{cast<D>(get<Is>(s))...};

// FAST: single builtin call
return __builtin_convertvector(s, vector_t<D, size<S>()>);
```

### `__builtin_shufflevector` for vector slice/concat

Replaces element-by-element `make_vector(get<Is>(c)...)` with a single shuffle:

```cpp
// SLOW: N-element braced init
return make_vector(get<Is>(c)...);

// FAST: single shuffle (returns GCC-style vector, bit_cast to ext_vector_type)
using R = vector_t<scalar_type, sizeof...(Is)>;
return __builtin_bit_cast(R, __builtin_shufflevector(c, c, Is...));
```

## 4. Avoid Intermediate Type Creation

### Bypass `concat_tuple` with direct indexing

`concat_tuple` creates intermediate tuple types when concatenating >4 tuples. Replace with direct per-element computation:

```cpp
// unfold_x_stride: instead of concat_tuple(per_group_results...)
// compute each element's stride directly via unfold_x_stride_at<J>()

// pickup_shape: instead of concat_tuple(conditional<match, tuple<T>, tuple<>>{}...)
// build a filtered index sequence, then make_tuple(get<filtered_indices>(Shape{})...)

// flatten_tuple: instead of concat_tuple(explode_tuple(get<Is>(t))...)
// directly index as get<local>(get<group>(t)) via flatten_at<T, J, GS>()
```

### Specify return type explicitly to avoid `std::common_type`

```cpp
// SLOW: triggers recursive std::common_type<D, D, D, ..., D> with 64 types
return vector_return_type<void, decltype(cast<D>(get<Is>(s)))...>{...};

// FAST: D is already known, skip common_type entirely
return vector_return_type<D, decltype(cast<D>(get<Is>(s)))...>{...};
```

### Add fold-expression fast paths for common patterns

```cpp
// reduce_tuple_mul for tuple<number<>...>: fold expression instead of recursive reduction
template<typename... Ns, std::enable_if_t<(is_constant_v<Ns> && ...), bool> = true>
constexpr auto reduce_tuple_mul(const tuple<Ns...>&) { return tuple<number<(Ns::value * ...)>>{}; }
```

## 5. Parallel Compilation

### Split device test files by template-instantiation cost

One file with 14 MFMA template instantiations (~3.9s) bottlenecks parallel builds. Split into per-type files (f16/f32/f8) to balance workload:

```
test_mfma.cu (3.9s) -> test_mfma_f16.cu (0.9s) + test_mfma_f32.cu (0.5s) + test_mfma_f8.cu (0.9s)
```

### Use `hipcc --genco` for device-only compilation when launching from Python

Eliminates the host pass entirely. Python loads the `.hsaco` via `hipModuleLoad` and launches with `hipModuleLaunchKernel` (HIP driver API).

## Compile-Time Measurement

### Use `-ftime-trace` for profiling

```bash
hipcc kernel.cc --cuda-device-only -c -o /dev/null \
  -Xclang -ftime-trace=trace.json
```

Analyze with chrome://tracing or a script:

```python
import json
with open('trace.json') as f: data = json.load(f)
events = data.get('traceEvents', data)
inst = [(e['dur'], e['args']['detail']) for e in events
        if e.get('name') == 'InstantiateFunction' and 'dur' in e]
inst.sort(key=lambda x: -x[0])
for dur, name in inst[:20]:
    print(f"{dur/1000:8.1f}ms  {name[:100]}")
```

### Key metrics to track

- **Function instantiations**: total count and per-function time
- **Frontend vs Backend**: frontend = template instantiation, backend = LLVM optimizer + codegen
- **Critical path**: the single slowest template chain determines wall-clock time

## Summary Table

| Technique | Typical savings | Where applied |
|-----------|----------------|---------------|
| Runtime `for` loops in load/store/MMA | 30-60% frontend | `buffer_view::load/store`, `tiled_mma_adaptor::operator()` |
| Runtime `flat_to_coords` | 40-50% frontend | `layout_to_offsets` |
| `__builtin_convertvector` | 5-10% frontend | `cast` for vectors >16 elements |
| `__builtin_shufflevector` | 3-5% frontend | `slice_impl` for vectors |
| Cache constexpr members | 10-15% frontend | `layout_load_traits`, `mma_a/b/c_len` |
| Direct indexing (bypass concat_tuple) | 5-10% frontend | `unfold_x_stride`, `pickup_shape`, `flatten_tuple` |
| `__HIP_DEVICE_COMPILE__` guard | ~50% per-file | All `.cu`/`.hip` files using opus.hpp |
| `-D__HIPCC_RTC__` | ~25% per-file | Compiler flags |
| `hipcc --genco` | ~15% per-file | Python-launched kernels |
| Split large TU files | Better parallelism | Test suites, multi-kernel builds |
