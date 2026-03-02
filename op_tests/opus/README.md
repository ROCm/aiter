# OPUS Tests

Unit tests for **OPUS** (AI Operator Micro Std). Contains a host-only C++ test and GPU device-kernel tests compiled into a shared library (`opus_device_test.so`) via `hipcc` and loaded from Python via `ctypes`.

OPUS headers live under `csrc/include/opus/`; all kernel code uses `#include "opus/opus.hpp"`.

## Folder structure

```
op_tests/opus/
├── test_opus_basic.cpp          # Host-only C++ test (no GPU)
├── build.sh                     # Builds test_opus_basic
├── device/                      # GPU kernel tests (hipcc -> .so, loaded via ctypes)
│   ├── test_mfma_f16.cu         # MFMA fp16/bf16 kernels
│   ├── test_mfma_f32.cu         # MFMA fp32 kernels
│   ├── test_mfma_f8.cu          # MFMA fp8/bf8 kernels
│   ├── test_mxfp.cu             # MXFP8/MXFP4 kernels (gfx950 only)
│   ├── test_vector_add.cu       # Vector addition kernel
│   ├── test_async_load.cu       # Async global->LDS->global copy kernel
│   ├── test_dtype_convert.cu    # FP32<->BF16/FP16/FP8/FP4 round-trip kernels
│   ├── test_load_store_if.cu    # Predicated load/store + free function API tests
│   ├── test_numeric_limits.cu   # opus::numeric_limits kernel
│   ├── test_mdiv.cu             # opus::magic_div kernel
│   ├── test_workgroup_barrier.cu# Workgroup barrier kernel
│   ├── setup.py                 # Parallel hipcc build: 11 .cu -> .o -> .so
│   └── test_opus_device.py      # Python test runner (builds .so, runs all tests)
├── run_tests.sh                 # Runs host test + device tests
└── README.md
```

## Running tests

Run both host and device tests with a single command:

```bash
./run_tests.sh   # host test + device tests (requires ROCm + PyTorch)
```

This builds and runs the C++ host test (`test_opus_basic`), then builds the device `.so` and runs all GPU kernel tests.

To run them individually:

```bash
./build.sh --test                  # host-only C++ test (no GPU needed)
python3 device/test_opus_device.py # device kernel tests only
```

## Build-time optimizations

The device test build applies several techniques from the OPUS best-practices guide
(`csrc/include/opus/README.md`) to minimize `hipcc` compile time:

| # | Optimization | Build time | `run_tests.sh` | Speedup |
|---|---|---:|---:|---:|
| 0 | Original (PyBind11 ext, sequential hipcc) | 18,400 ms | ~20 s | 1.0x |
| 1 | Replace PyBind11/torch with ctypes + `extern "C"` | 12,500 ms | ~14 s | 1.5x |
| 2 | Parallel compilation (each .cu -> .o, then link) | 3,950 ms | ~9 s | 4.7x |
| 3 | Split `test_mfma.cu` into 3 files (f16/f32/f8) | 2,670 ms | ~8 s | 6.9x |
| 4 | Add `-D__HIPCC_RTC__` to reduce device-pass header parsing | 2,070 ms | ~7.5 s | 8.9x |
| 5 | Replace `<hip/hip_runtime.h>` with `hip_host_minimal.h` | **1,740 ms** | **~6.9 s** | **10.5x** |

### What each optimization does

1. **ctypes instead of PyBind11** — The old `opus_device_test_ext.cpp` (490 lines)
   included `<torch/extension.h>` and `<pybind11/pybind11.h>`, adding ~100K lines of
   C++ template headers to every compilation. Replaced with plain `extern "C"` host
   launchers in `.cu` files, loaded via `ctypes.CDLL` in Python.

2. **Parallel compilation** — `setup.py` compiles each `.cu` file to a `.o` in parallel
   using `ProcessPoolExecutor`, then links all `.o` files into a single `.so`. This
   turns the build from sequential (sum of all file times) into parallel (bounded by
   the slowest file).

3. **MFMA file splitting** — The original `test_mfma.cu` had 14 template instantiations
   and took ~3.9s alone (the parallel build bottleneck). Split into `test_mfma_f16.cu`,
   `test_mfma_f32.cu`, and `test_mfma_f8.cu` to balance the workload across cores.

4. **`-D__HIPCC_RTC__`** — This flag tells the HIP headers to skip declarations not
   needed for runtime compilation, reducing preprocessing load on both host and device
   passes. Saves ~500ms per file on header parsing.

5. **`hip_host_minimal.h`** — A ~70-line header (`csrc/include/hip_host_minimal.h`) that
   declares only the dozen HIP APIs the host launcher code actually uses (`dim3`,
   `hipLaunchKernelGGL`, `hipMalloc`, `hipDeviceSynchronize`, etc.), replacing
   `<hip/hip_runtime.h>` (~100K preprocessed lines) on the host pass. Combined with
   `-D__HIPCC_RTC__` for the device pass, this eliminates nearly all unnecessary header
   parsing.

### Current time breakdown (`time ./run_tests.sh`)

```
real    0m6.9s      (wall clock)
user    0m18.4s     (CPU time, summed across parallel hipcc jobs)
sys     0m2.1s

Phase                             Time
────────────────────────────────  ──────
Host build (hipcc test_opus_basic)  1,400 ms
Host run (13 unit tests)               12 ms
Device .so build (11 .cu, parallel) 1,740 ms
  compile (11 parallel jobs)          1,710 ms
  link (.o -> .so)                       30 ms
Device tests (torch import + GPU)   3,900 ms
  torch import                        1,600 ms
  kernel execution (50+ tests)        2,300 ms
────────────────────────────────  ──────
Total wall clock                    ~6.9 s
```

### Per-file device compile times

```
test_vector_add.cu         187 ms
test_async_load.cu         185 ms
test_numeric_limits.cu     191 ms
test_workgroup_barrier.cu  216 ms
test_mdiv.cu               243 ms
test_mxfp.cu               248 ms
test_load_store_if.cu      354 ms
test_dtype_convert.cu      506 ms
test_mfma_f32.cu         1,445 ms
test_mfma_f8.cu          1,654 ms
test_mfma_f16.cu         1,712 ms  <-- critical path
link                        31 ms
```

## How to add a new device test

All GPU kernel tests live in `device/` and are compiled into `opus_device_test.so`.
To add a new kernel test (e.g. `my_kernel`):

### 1. Create the kernel source

Add `device/test_my_kernel.cu`. The file uses conditional compilation to separate
device and host passes:

```cpp
// device/test_my_kernel.cu
#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass: use opus.hpp + builtins, no hip_runtime.h ──
#include "opus/opus.hpp"

__global__ void my_kernel(const float* in, float* out, int n) {
    int tid = __builtin_amdgcn_workitem_id_x()
            + __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x();
    if (tid < n) out[tid] = in[tid] * 2.0f;
}

#else
// ── Host pass: minimal HIP header for kernel launch API ──
// #include <hip/hip_runtime.h>   // replaced by hip_host_minimal.h for faster builds
#include "hip_host_minimal.h"

__global__ void my_kernel(const float* in, float* out, int n);

extern "C" void run_my_kernel(const void* d_in, void* d_out, int n) {
    dim3 grid((n + 255) / 256), block(256);
    hipLaunchKernelGGL(my_kernel, grid, block, 0, nullptr,
                       (const float*)d_in, (float*)d_out, n);
    hipDeviceSynchronize();
}
#endif
```

### 2. Register the source in setup.py

Add the filename to `_CU_SOURCES` in `device/setup.py`:

```python
_CU_SOURCES = [
    ...
    "test_my_kernel.cu",
]
```

### 3. Add the ctypes wrapper and Python test

In `device/test_opus_device.py`:

1. Add a method to the `OpusDeviceLib` class:
   ```python
   def run_my_kernel(self, In, Out, n):
       fn = self._lib.run_my_kernel
       fn.restype = None
       fn.argtypes = [_VP, _VP, _I]
       fn(self._ptr(In), self._ptr(Out), n)
   ```

2. Add a test function and call it from `main()`:
   ```python
   def test_my_kernel(mod):
       # Create input tensors, call mod.run_my_kernel(...), compare with reference
       ...
   ```

### 4. Verify

```bash
./run_tests.sh
```

## Device test summary

| Test | Variant | OPUS APIs exercised | Arch |
|---|---|---|---|
| `test_mfma_f16` | 32x32x8 fp16/bf16 | `make_tiled_mma`, `mfma_adaptor_swap_ab`, `partition_layout_a/b/c`, `make_gmem`, `cast` | gfx942 |
| `test_mfma_f16` | 16x16x16 fp16/bf16 | (same as above) | gfx942 |
| `test_mfma_f16` | 32x32x16 fp16/bf16 | (same, uses base 32x32x8 with K-loop on gfx942; native on gfx950) | gfx942 + gfx950 |
| `test_mfma_f16` | 16x16x32 fp16/bf16 | (same, uses base 16x16x16 with K-loop on gfx942; native on gfx950) | gfx942 + gfx950 |
| `test_mfma_f32` | 32x32x2 f32 | `make_tiled_mma`, `partition_layout_a/b/c`, `make_gmem` | gfx942 |
| `test_mfma_f32` | 16x16x4 f32 | (same as above) | gfx942 |
| `test_mfma_f8` | 32x32x16 fp8/bf8 | `make_tiled_mma`, `partition_layout_a/b/c`, `make_gmem` (fp32 output, no cast) | gfx942 + gfx950 |
| `test_mfma_f8` | 16x16x32 fp8/bf8 | (same as above) | gfx942 + gfx950 |
| `test_mxfp` | mxfp8_32x32x64 | `mfma<fp8_t,fp8_t,fp32_t,32,32,64>` (scaled overload), direct data-layout load/store | gfx950 |
| `test_mxfp` | mxfp8_16x16x128 | `mfma<fp8_t,fp8_t,fp32_t,16,16,128>` (scaled overload) | gfx950 |
| `test_mxfp` | mxfp4_32x32x64 | `mfma<fp4_t,fp4_t,fp32_t,32,32,64>` (scaled overload), fp4x2 packed nibble handling | gfx950 |
| `test_mxfp` | mxfp4_16x16x128 | `mfma<fp4_t,fp4_t,fp32_t,16,16,128>` (scaled overload) | gfx950 |
| `test_vector_add` | — | `make_gmem`, vectorized `load<N>` / `store<N>` | all |
| `test_async_load` | — | `make_gmem`, `gmem::async_load`, `s_waitcnt_vmcnt` | all |
| `test_dtype_convert` | fp32<->bf16 scalar | `cast<bf16_t>(fp32_t)` RNE (explicit `0_I` on gfx942, hw default on gfx950) | all |
| `test_dtype_convert` | fp32<->bf16 x4 vec | `cast<bf16_t>(fp32x4_t)` generic vectorized | all |
| `test_dtype_convert` | fp32<->fp16 scalar | `cast<fp16_t>(fp32_t)` / `cast<fp32_t>(fp16_t)` | all |
| `test_dtype_convert` | fp32<->fp16 x4 vec | `cast<fp16_t>(fp32x4_t)` generic vectorized | all |
| `test_dtype_convert` | fp32<->fp8 scalar | `cast<fp8_t>(fp32_t)` via `cvt_pk_fp8_f32` lo / `cvt_f32_fp8` | gfx942 + gfx950 |
| `test_dtype_convert` | fp32<->fp8 x2 pk | `cast<fp8_t>(fp32x2_t)` packed x2 | gfx942 + gfx950 |
| `test_dtype_convert` | fp32<->fp8 x4 pk | `cast<fp8_t>(fp32x4_t)` packed x4 | gfx942 + gfx950 |
| `test_dtype_convert` | fp32<->fp8 x8 fold | `cast<fp8_t>(fp32x8_t)` auto-fold 2x4 + `unfold_from_container` | gfx942 + gfx950 |
| `test_dtype_convert` | fp32<->fp4 x2 pk | `cast<fp4_t>(fp32x2_t)` packed x2, e2m1 | gfx950 |
| `test_dtype_convert` | fp32<->fp4 x4 pk | `cast<fp4_t>(fp32x4_t)` packed x4, e2m1 | gfx950 |
| `test_dtype_convert` | fp32<->fp4 x8 pk | `cast<fp4_t>(fp32x8_t)` packed x8, e2m1 | gfx950 |
| `test_load_store_if` | predicated_copy | `gmem::load_if`, `gmem::store_if`, free functions `opus::load_if`/`opus::store_if`, `layout_linear::operator+` | all |
| `test_load_store_if` | free_func_vector_add | Free functions `opus::load`/`opus::store`, `is_gmem_v`/`is_mem_v` type traits | all |
| `test_load_store_if` | predicated_async_load | `gmem::async_load_if`, free function `opus::async_load_if`, `layout_linear::operator+` | all |
| `test_numeric_limits` | all types | `opus::numeric_limits<T>` for fp32/fp16/bf16/fp8/bf8/i32/i16/i8/u8 | all |
| `test_mdiv` | 11 divisors | `opus::magic_div` integer division by magic multiply | all |
| `test_workgroup_barrier` | cumulative + streamk | `opus::workgroup_barrier` cross-workgroup synchronization | all |

Total: **50+ test calls** (14 MFMA + 4 MXFP + 1 vector_add + 1 async_load + 11 dtype_convert + 3 load_store_if + 9 numeric_limits + 11 mdiv + 4 workgroup_barrier).

## Notes

- The extension compiles with `--offload-arch=<detected>` (see `device/setup.py`) to target only the current GPU and speed up builds.
- MFMA tests are runtime-gated by GPU architecture (`gcnArchName`). Tests for unsupported architectures are automatically skipped.
  - 32x32x8 and 16x16x16 variants: gfx942 only.
  - 32x32x16 and 16x16x32 fp16/bf16 variants: gfx942 (via step-K decomposition) + gfx950 (native instruction).
  - 32x32x16 and 16x16x32 fp8/bf8 variants: gfx942 + gfx950 (native instruction on both). Output is raw fp32 accumulator.
- **BF16 rounding**: `opus::cast<bf16_t>` default rounding mode differs by architecture:
  - gfx942: default is truncation (rm=2). Pass `0_I` as 2nd argument to select round-to-nearest-even (RNE).
  - gfx950: default is already RNE (hardware). No 2nd argument needed.
  The dtype_convert bf16 test and MFMA bf16 tests both use RNE so that the kernel result matches PyTorch `.to(bfloat16)`.
- FP8 = `float8_e4m3fnuz` (gfx942) / `float8_e4m3fn` (gfx950), BF8 = `float8_e5m2fnuz` (gfx942) / `float8_e5m2` (gfx950).
- FP4 = E2M1 (4-bit: 1 sign, 2 exponent, 1 mantissa). Representable values: +/-{0, 0.5, 1, 1.5, 2, 3, 4, 6}. gfx950 only.
- **MXFP** (unified into `struct mfma`, scaled `operator()` overload): gfx950-only `__builtin_amdgcn_mfma_scale_f32_{32x32x64,16x16x128}_f8f6f4` intrinsics. Support MXFP8 (fp8*fp8) and MXFP4 (fp4*fp4) with E8M0 block exponent scaling. Tests use `scale=127` (2^0=1.0, no scaling) and verify `C = A @ B` (standard matmul, **not** swap_ab). The data layout follows the CDNA4 Matrix Core specification.
- `test_opus_device.py` does a fresh build on every run (cleans previous `.so`) to ensure changes are picked up.
