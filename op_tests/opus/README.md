# OPUS Tests

Unit tests for **OPUS** (AI Operator Micro Std). Contains a host-only C++ test and GPU device-kernel tests exposed as a single **PyTorch extension** (`opus_device_test`) and run from Python.

OPUS headers live under `csrc/include/opus/`; all kernel code uses `#include "opus/opus.hpp"`.

## Folder structure

```
op_tests/opus/
├── test_opus_basic.cpp          # Host-only C++ test (no GPU)
├── build.sh                     # Builds test_opus_basic
├── device/                      # GPU kernel tests (single PyTorch extension)
│   ├── test_mfma.cu             # MFMA 32x32x8 fp16 kernel (gfx942 only)
│   ├── test_mfma.h              # C API header for MFMA
│   ├── test_vector_add.cu       # Vector addition kernel using OPUS gmem
│   ├── test_vector_add.h        # C API header for vector_add
│   ├── test_async_load.cu       # Async global->LDS->global copy kernel
│   ├── test_async_load.h        # C API header for async_load
│   ├── test_dtype_convert.cu    # FP32<->BF16/FP16/FP8 round-trip kernels
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

| Test | OPUS APIs exercised | Arch |
|---|---|---|
| `test_mfma` | `make_mfma`, `mfma_adaptor_swap_ab`, `make_tiled_mma`, `partition_layout_a/b/c`, `make_gmem` | gfx942 only |
| `test_vector_add` | `make_gmem`, vectorized `load<VECTOR_SIZE>` / `store<VECTOR_SIZE>` | all |
| `test_async_load` | `make_gmem`, `gmem::async_load`, `s_waitcnt_vmcnt` | all |
| `test_dtype_convert` (fp32<->bf16) | `fp32_to_bf16` (truncation mode), `bf16_to_fp32` | all |
| `test_dtype_convert` (fp32<->fp16) | `fp32_to_fp16`, `fp16_to_fp32` | all |
| `test_dtype_convert` (fp32<->fp8) | `cast<fp8_t>(fp32x4_t)`, `cast<fp32_t>(fp8x4_t)` (packed x4) | all |

## Notes

- The extension compiles with `--offload-arch=native` (see `device/setup.py`) to target only the current GPU and speed up builds.
- MFMA tests require `gfx942` (MI300); they are automatically skipped on other architectures.
- `test_opus_device.py` does a fresh build on every run (cleans previous `.so` and `build/` dir) to ensure changes are picked up.
