---
name: aiter-add-operator
description: >
 End-to-end recipe for adding a brand-new operator to AITER. Walks through the
 5 required artifacts: (1) the HIP/CK/Triton kernel under `csrc/` or
 `aiter/ops/triton/`, (2) the pybind module under `csrc/pybind/`, (3) the JIT
 entry in `aiter/jit/optCompilerConfig.json`, (4) the Python wrapper in
 `aiter/ops/.py` using `@compile_ops`, and (5) the correctness test
 under `op_tests/test_.py`. Use when a user says "add op", "new operator",
 "register a kernel in AITER", or requests a missing op.
 Usage: /aiter-add-operator 
allowed-tools: Bash Read Edit Grep Glob
---

# Add a New Operator to AITER

AITER operators are composed of 5 tightly-coupled pieces. Miss any of them and
the op silently fails to load. This skill walks through each piece using a
running example: adding `my_op(input, out)` backed by a HIP kernel.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| ` ` | Yes | snake_case name of the new op, e.g. `fused_silu_quant`. |
| `--backend` | No | `hip` (default), `ck`, `triton`, or `flydsl`. |

If `` is not provided, ask the user.

## Mental model

```
  Python call site (user code)
    |
    v
  aiter/ops/my_op.py        <-- 4. Python wrapper with @compile_ops("module_my_op")
    |
    v
  aiter/jit/optCompilerConfig.json  <-- 3. module_my_op -> srcs, flags, blob_gen_cmd
    |
    v (JIT compile)
  csrc/pybind/my_op_pybind.cu       <-- 2. PyBind registration
  csrc/kernels/my_op_kernels.cu      <-- 1. HIP kernel (or csrc/ck_my_op/ for CK, aiter/ops/triton/... for Triton)
    |
    v
  aiter/jit/module_my_op.so  (generated)

  op_tests/test_my_op.py            <-- 5. Correctness/perf test
```

All 5 files MUST exist. The JIT system looks up `MD_NAME = "module_my_op"` in
`optCompilerConfig.json`; a missing entry produces a cryptic
`ModuleNotFoundError: No module named 'module_my_op'`.

---

## Step 1: Write the kernel

### Option A: HIP kernel (most ops)

Path: `csrc/kernels/my_op_kernels.cu`

```cpp
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include 
#include "aiter_hip_common.h"

template 
__global__ void my_op_kernel(const T* __restrict__ input,
                              T* __restrict__ output,
                              int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  output[idx] = input[idx] * input[idx];
}

void my_op(torch::Tensor& out, torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(), "tensors must be on GPU");
  TORCH_CHECK(input.numel() == out.numel(), "size mismatch");

  const int n = input.numel();
  const int block = 256;
  const int grid = (n + block - 1) / block;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "my_op", [&] {
        my_op_kernel<<>>(
            input.data_ptr(), out.data_ptr(), n);
      });
}
```

Also declare the function in a header so the pybind file can include it:

`csrc/include/my_op.h`:

```cpp
#pragma once
#include 
void my_op(torch::Tensor& out, torch::Tensor& input);
```

### Option B: CK kernel

Put all generated CK instances under `csrc/ck_my_op/` and follow the pattern of
`csrc/ck_gemm_a8w8/` (see the `aiter-ck-tune` skill for the tuning flow). Wire
the directory in via `blob_gen_cmd` in `optCompilerConfig.json`.

### Option C: Triton kernel

Use `aiter/ops/triton/` with the layout described by the
`aiter-triton-kernel` skill. No `optCompilerConfig.json` entry needed — Triton
kernels are JIT'd by Triton itself.

---

## Step 2: PyBind registration

Path: `csrc/pybind/my_op_pybind.cu`

```cpp
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include 
#include "my_op.h"

PYBIND11_MODULE(module_my_op, m) {
  m.def("my_op", &my_op, "Squares every element of input into out.",
        py::arg("out"), py::arg("input"));
}
```

The module name in `PYBIND11_MODULE(module_my_op, ...)` **must match** the key
used in `optCompilerConfig.json` AND the argument to `@compile_ops` in the
Python wrapper.

---

## Step 3: JIT config entry

Add a new key to `aiter/jit/optCompilerConfig.json`:

```json
{
  "module_my_op": {
    "srcs": [
      "f'{AITER_CSRC_DIR}/pybind/my_op_pybind.cu'",
      "f'{AITER_CSRC_DIR}/kernels/my_op_kernels.cu'"
    ],
    "flags_extra_cc": [],
    "flags_extra_hip": [
      "'-ffast-math'"
    ],
    "extra_ldflags": "None",
    "extra_include": [
      "f'{AITER_CSRC_DIR}/include'"
    ],
    "verbose": "False",
    "blob_gen_cmd": "''"
  }
}
```

**Notes**:
- Values are f-string source code that `core.py` `eval()`s — do NOT use
  regular strings. Wrap literals in `'...'`.
- Keep `torch_exclude: "True"` only for enum/tensor-metadata-only modules
  (see `module_aiter_enum`).
- Set `hipify: "False"` if your code is already HIP (not CUDA) — the default
  hipify pass can mangle ROCm-specific intrinsics (see `module_pa_ragged`).
- For CK-tile integration, add `blob_gen_cmd` with a generator script (see
  `csrc/ck_gemm_a8w8/gen_instances.py` as reference).

---

## Step 4: Python wrapper

Path: `aiter/ops/my_op.py`

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops

MD_NAME = "module_my_op"


@compile_ops("module_my_op")
def my_op(out: Tensor, input: Tensor) -> None: ...
```

Then re-export from `aiter/__init__.py` if the op should be user-visible:

```python
from .ops.my_op import my_op as my_op
```

**How `@compile_ops` works**:
- On first call, it looks up `"module_my_op"` in
  `aiter/jit/optCompilerConfig.json`, JIT-compiles the `.so`, and swaps the
  stub body for the compiled C++ function.
- The Python function body (`...`) is never executed — it's just a type
  signature for IDEs / static checkers.
- Arguments must match the pybind signature order and types.

---

## Step 5: Correctness test

Path: `op_tests/test_my_op.py`

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import torch
from aiter.ops.my_op import my_op
from aiter.test_common import checkAllclose, perftest
from aiter import dtypes


@perftest()
def run_aiter(input):
    out = torch.empty_like(input)
    my_op(out, input)
    return out


def ref_my_op(input):
    return input * input


def test_my_op(dtype, n):
    x = torch.randn(n, dtype=dtype, device="cuda")
    expected = ref_my_op(x)
    got, latency = run_aiter(x)
    checkAllclose(got, expected, rtol=1e-3, atol=1e-3,
                   msg=f"my_op dtype={dtype} n={n} {latency:.1f}us")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dtype", default=None)
    ap.add_argument("-n", type=int, default=None)
    args = ap.parse_args()

    l_dtype = [torch.float16, torch.bfloat16, torch.float32]
    l_n = [1024, 16 * 1024, 1024 * 1024]

    if args.dtype:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.n:
        l_n = [args.n]

    for d in l_dtype:
        for n in l_n:
            test_my_op(d, n)
```

Run it:

```bash
python3 op_tests/test_my_op.py
AITER_LOG_MORE=1 python3 op_tests/test_my_op.py   # verbose JIT logs
AITER_REBUILD=1 python3 op_tests/test_my_op.py     # force rebuild
```

The first run compiles `module_my_op.so` into `aiter/jit/`. Subsequent runs
reuse it until source files change.

---

## Step 6: Register with the test runner (CI)

AITER's CI discovers tests under `op_tests/*.py` automatically (see
`.github/scripts/aiter_test.sh`). No additional registration is required as
long as the file is named `test_.py` and lives at the top level
of `op_tests/`.

Triton tests go under `op_tests/triton_tests//test_.py`
and are run with `pytest`.

Multi-GPU tests go under `op_tests/multigpu_tests/`.

---

## Checklist

Before opening a PR:

- [ ] `csrc/kernels//my_op_kernels.cu` (or equivalent) compiles cleanly.
- [ ] `csrc/pybind/my_op_pybind.cu` binds the module with the exact same name as the JSON key.
- [ ] `aiter/jit/optCompilerConfig.json` has a `module_my_op` entry with correct `srcs` / `flags_extra_hip` / `extra_include`.
- [ ] `aiter/ops/my_op.py` uses `@compile_ops("module_my_op")` and the signature matches pybind.
- [ ] `op_tests/test_my_op.py` runs standalone and passes.
- [ ] `AITER_REBUILD=1 python3 op_tests/test_my_op.py` produces `aiter/jit/module_my_op.so`.
- [ ] `ruff check aiter/ op_tests/` and `black aiter/ op_tests/` pass (see `format-code` skill).
- [ ] `clang-format-18 -i csrc/pybind/my_op_pybind.cu csrc/kernels/my_op_kernels.cu`.
- [ ] PR title follows the `[Feature]` / `[Kernel]` convention from `CONTRIBUTE.md`.

## Common pitfalls

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'module_my_op'` | Missing entry in `optCompilerConfig.json`, or typo between the JSON key, `PYBIND11_MODULE(...)`, and `@compile_ops(...)`. All three must match. |
| `undefined symbol: my_op(at::Tensor&, at::Tensor&)` | Function declared in header but not defined in one of the `srcs`. Add the .cu to `srcs`. |
| `hipErrorNoBinaryForGpu` | Kernel was compiled for the wrong arch. Rebuild with `GPU_ARCHS="gfx942;gfx950" AITER_REBUILD=1 ...`. |
| Test passes with dtype=f32 but fails with bf16 | AT_DISPATCH missing the BF16 branch; use `AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16, ...)`. |
| CK kernel compile timeout | Split instances into multiple `*_part{1,2,3}.cpp` files as in `device_gemm_multiply_multiply_xdl_*`. |
| The JIT rebuilds every invocation | Source mtime in a shared volume is unstable — set `AITER_REBUILD=0` and check for clock skew. |

## Reference operators to copy

| Backend | Simplest example to copy |
|---------|--------------------------|
| HIP scalar | `module_activation` (`aiter/ops/activation.py`, `csrc/kernels/activation_kernels.cu`). |
| HIP + CK integration | `module_pa_ragged` (`aiter/ops/attention.py`, `csrc/kernels/attention_ragged.cu`). |
| CK GEMM with tuning | `module_gemm_a8w8` + `module_gemm_a8w8_tune` (see `csrc/ck_gemm_a8w8/README.md`). |
| Triton kernel | `aiter/ops/triton/gemm/basic/gemm_a16w16.py`. |
| FlyDSL kernel | `aiter/ops/flydsl/` (optional; A4W4 MoE). |
