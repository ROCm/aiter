# OPUS C++ Tests

C++ tests for **OPUS** (AI Operator Micro Std) in this repo: a host-only test and a GPU MFMA test exposed as a **PyTorch extension** and run from Python.

## Test structure

| Location | Component | Description |
|----------|-----------|-------------|
| (root) | `test_opus_basic.cpp` | **Host test** (standalone). Built by `build.sh` into `test_opus_basic`. Covers numbers/sequences, arrays, tuples, `static_for`, type traits, layouts. No GPU. |
| **`mfma/`** | `test_opus_mfma.cpp` | **Kernel-only** HIP code: 32×32×8 fp16 MFMA using OPUS `make_mfma`, `make_gmem`, `partition_layout_*`. No `main()`; used only by the extension. |
| **`mfma/`** | `test_opus_mfma.h` | Declares `run_mfma_32x32x8_f16()` (C linkage) for the extension. |
| **`mfma/`** | `opus_mfma_ext.cpp` | PyTorch C++ extension (pybind11): wraps `run_mfma_32x32x8_f16` and exposes `run_mfma_32x32x8_f16(A, B, C)` for `torch.float16` CUDA tensors (A 32×8, B 32×8, C 32×32). |
| **`mfma/`** | `test_opus_mfma.py` | Builds the extension via `torch.utils.cpp_extension.load` (JIT), runs with random A/B, compares result to `torch` GEMM (A @ B.T). |

OPUS headers live under repo `csrc/include/`; the MFMA code uses `#include "opus/opus.hpp"` with include path set from `mfma/` to `../../../csrc/include`.

## Building and running

- **C++ host test only**
  ```bash
  ./build.sh          # compile test_opus_basic
  ./build.sh --test   # compile and run test_opus_basic
  ./build.sh --clean  # remove test_opus_basic
  ```

- **MFMA test (PyTorch extension, needs PyTorch + ROCm/CUDA)** — run from `op_tests/opus`:
  ```bash
  python3 mfma/test_opus_mfma.py   # JIT-builds extension, runs test vs torch GEMM
  ```

`build.sh` uses `hipcc` (or `HIPCC`). The Python test uses `torch.utils.cpp_extension.load()` and needs PyTorch built for the same backend (ROCm or CUDA).

## Running all tests

From `op_tests/opus` you can run the full suite with:

```bash
./run_tests.sh
```

This runs the C++ host test (`./build.sh --test`) then the MFMA PyTorch test (`mfma/test_opus_mfma.py`). Requires `hipcc` and (for the MFMA test) PyTorch + ROCm/CUDA.

## Running in Docker

To run the same suite inside a ROCm container:

```bash
./run_tests_in_docker.sh
```

This starts the container, `cd`s to `op_tests/opus`, and runs `./run_tests.sh` inside it. The image must have PyTorch and ROCm (e.g. `rocm/atom`).

## Summary

- **Host:** `test_opus_basic` — OPUS core types and layout (standalone executable).
- **Device (all under `mfma/`):** MFMA kernel in `test_opus_mfma.cpp` → PyTorch extension in `opus_mfma_ext.cpp` → tested by `test_opus_mfma.py` (random A/B, compare to torch).
