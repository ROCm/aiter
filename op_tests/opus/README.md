# OPUS Tests

Unit tests for **OPUS** (AI Operator Micro Std). Contains a host-only C++ test and GPU device-kernel tests exposed as a single **PyTorch extension** (`opus_device_test`) and run from Python.

OPUS headers live under `csrc/include/opus/`; all kernel code uses `#include "opus/opus.hpp"`.

## Folder structure

```
op_tests/opus/
├── test_opus_basic.cpp          # Host-only C++ test (no GPU)
├── build.sh                     # Builds test_opus_basic
├── device/                      # GPU kernel tests (single PyTorch extension)
│   ├── test_mfma.cu             # MFMA kernels: fp16/bf16/fp8/bf8 variants
│   ├── test_mfma.h              # C API header for MFMA
│   ├── test_mxfp.cu             # MXFP8/MXFP4 kernels: fp8/fp4 (gfx950 only)
│   ├── test_mxfp.h              # C API header for MXFP
│   ├── test_load_store_if.cu    # Predicated load/store + free function API tests
│   ├── test_load_store_if.h     # C API header for load_store_if
│   ├── test_vector_add.cu       # Vector addition kernel using OPUS gmem
│   ├── test_vector_add.h        # C API header for vector_add
│   ├── test_async_load.cu       # Async global->LDS->global copy kernel
│   ├── test_async_load.h        # C API header for async_load
│   ├── test_dtype_convert.cu    # FP32<->BF16/FP16/FP8/FP4 round-trip kernels
│   ├── test_dtype_convert.h     # C API header for dtype_convert
│   ├── opus_device_test_ext.cpp # Pybind module: binds all device kernels
│   ├── setup.py                 # Builds the opus_device_test extension
│   └── test_opus_device.py      # Python test: runs all device kernel tests
├── run_tests.sh                 # Runs host test + device tests
├── run_tests_in_docker.sh       # Runs run_tests.sh inside a ROCm Docker container
└── README.md
```

## Building and running

### Host-only test

```bash
./build.sh          # compile test_opus_basic
./build.sh --test   # compile and run
./build.sh --clean  # remove binary
```

### Device kernel tests (requires PyTorch + ROCm)

```bash
python3 device/test_opus_device.py
```

This builds the `opus_device_test` PyTorch extension (via `setup.py`) and runs all device tests.

### Full suite

```bash
./run_tests.sh              # host test + device tests
./run_tests_in_docker.sh    # same, inside a ROCm Docker container
```

## How to add a new device test

All GPU kernel tests live in `device/` and are compiled into a single PyTorch extension (`opus_device_test`). To add a new kernel test (e.g. `my_kernel`):

### 1. Create the kernel source

Add two files in `device/`:

- **`test_my_kernel.cu`** -- HIP kernel + host launcher function. Use `extern "C"` for the launcher so it can be called from the pybind module.

```cpp
// device/test_my_kernel.cu
#include <hip/hip_runtime.h>
#include "opus/opus.hpp"
#include "test_my_kernel.h"

__global__ void my_kernel(...) {
    // Use OPUS APIs: make_gmem, make_tiled_mma, partition_layout_*, etc.
}

extern "C" void run_my_kernel(const void* d_in, void* d_out, int n) {
    // Launch kernel, check errors, synchronize
}
```

- **`test_my_kernel.h`** -- C-linkage header declaring the launcher.

```cpp
// device/test_my_kernel.h
#ifndef OP_TESTS_OPUS_DEVICE_TEST_MY_KERNEL_H
#define OP_TESTS_OPUS_DEVICE_TEST_MY_KERNEL_H
#ifdef __cplusplus
extern "C" {
#endif

void run_my_kernel(const void* d_in, void* d_out, int n);

#ifdef __cplusplus
}
#endif
#endif
```

### 2. Add the pybind wrapper

In `device/opus_device_test_ext.cpp`:

1. Include the new header:
   ```cpp
   #include "test_my_kernel.h"
   ```

2. Add a torch wrapper function that validates tensor properties and calls the C launcher:
   ```cpp
   static void run_my_kernel_torch(torch::Tensor In, torch::Tensor Out) {
       TORCH_CHECK(In.is_cuda(), "In must be a CUDA tensor");
       // ... more checks ...
       run_my_kernel(In.data_ptr(), Out.data_ptr(), static_cast<int>(In.numel()));
   }
   ```

3. Register it in the `PYBIND11_MODULE` block:
   ```cpp
   m.def("run_my_kernel", &run_my_kernel_torch, "Description of my_kernel");
   ```

### 3. Register the source in setup.py

Add `test_my_kernel.cu` to the `sources` list in `device/setup.py`:

```python
sources=[
    os.path.join(_THIS_DIR, "test_mfma.cu"),
    os.path.join(_THIS_DIR, "test_vector_add.cu"),
    os.path.join(_THIS_DIR, "test_async_load.cu"),
    os.path.join(_THIS_DIR, "test_dtype_convert.cu"),
    os.path.join(_THIS_DIR, "test_my_kernel.cu"),      # <-- add this
    os.path.join(_THIS_DIR, "opus_device_test_ext.cpp"),
],
```

### 4. Add the Python test

In `device/test_opus_device.py`, add a test function and call it from `main()`:

```python
def test_my_kernel(mod):
    """Test my_kernel."""
    # Create input tensors
    # Call mod.run_my_kernel(...)
    # Compare with reference
    # Return 0 on success, 1 on failure

def main():
    # ... existing code ...
    failures += test_my_kernel(mod)
```

### 5. Verify

```bash
./run_tests_in_docker.sh
```

All tests (including the new one) will build and run inside the Docker container.

## Device test summary

| Test | Variant | OPUS APIs exercised | Arch |
|---|---|---|---|
| `test_mfma` | 32x32x8 fp16/bf16 | `make_tiled_mma`, `mfma_adaptor_swap_ab`, `partition_layout_a/b/c`, `make_gmem`, `cast` | gfx942 |
| `test_mfma` | 16x16x16 fp16/bf16 | (same as above) | gfx942 |
| `test_mfma` | 32x32x16 fp16/bf16 | (same, uses base 32x32x8 with K-loop on gfx942; native on gfx950) | gfx942 + gfx950 |
| `test_mfma` | 16x16x32 fp16/bf16 | (same, uses base 16x16x16 with K-loop on gfx942; native on gfx950) | gfx942 + gfx950 |
| `test_mfma` | 32x32x16 fp8/bf8 | `make_tiled_mma`, `partition_layout_a/b/c`, `make_gmem` (fp32 output, no cast) | gfx942 + gfx950 |
| `test_mfma` | 16x16x32 fp8/bf8 | (same as above) | gfx942 + gfx950 |
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

Total: **37 test calls** (14 MFMA + 4 MXFP + 1 vector_add + 1 async_load + 11 dtype_convert + 3 load_store_if + 1 mdiv + 2 workgroup_barrier).

## Notes

- The extension compiles with `--offload-arch=native` (see `device/setup.py`) to target only the current GPU and speed up builds.
- MFMA tests are runtime-gated by GPU architecture (`gcnArchName`). Tests for unsupported architectures are automatically skipped.
  - 32x32x8 and 16x16x16 variants: gfx942 only.
  - 32x32x16 and 16x16x32 fp16/bf16 variants: gfx942 (via step-K decomposition) + gfx950 (native instruction).
  - 32x32x16 and 16x16x32 fp8/bf8 variants: gfx942 + gfx950 (native instruction on both). Output is raw fp32 accumulator.
- **BF16 rounding**: `opus::cast<bf16_t>` default rounding mode differs by architecture:
  - gfx942: default is truncation (rm=2). Pass `0_I` as 2nd argument to select round-to-nearest-even (RNE).
  - gfx950: default is already RNE (hardware). No 2nd argument needed.
  The dtype_convert bf16 test and MFMA bf16 tests both use RNE so that the kernel result matches PyTorch's `.to(bfloat16)`.
- FP8 = `float8_e4m3fnuz` (gfx942) / `float8_e4m3fn` (gfx950), BF8 = `float8_e5m2fnuz` (gfx942) / `float8_e5m2` (gfx950).
- FP4 = E2M1 (4-bit: 1 sign, 2 exponent, 1 mantissa). Representable values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. gfx950 only.
- **MXFP** (unified into `struct mfma`, scaled `operator()` overload): gfx950-only `__builtin_amdgcn_mfma_scale_f32_{32x32x64,16x16x128}_f8f6f4` intrinsics. Support MXFP8 (fp8\*fp8) and MXFP4 (fp4\*fp4) with E8M0 block exponent scaling. Tests use `scale=127` (2^0=1.0, no scaling) and verify `C = A @ B` (standard matmul, **not** swap\_ab). The data layout follows the CDNA4 Matrix Core specification.
- `test_opus_device.py` does a fresh build on every run (cleans previous `.so` and `build/` dir) to ensure changes are picked up.
