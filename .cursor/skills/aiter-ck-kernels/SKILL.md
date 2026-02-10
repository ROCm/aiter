---
name: aiter-ck-kernels
description: Write Composable Kernel (CK) C++ GPU kernels, tests, and build configurations for the aiter project. Use when creating or modifying CK-based GEMM, MHA, MoE, or normalization kernels that use AMD's Composable Kernel library. Covers CK template instantiation, codegen, pybind integration, JIT compilation, and the optCompilerConfig.json build system.
---

# Composable Kernel (CK) Kernels in Aiter

## Project Layout

| Component | Path |
|-----------|------|
| CK kernel source | `csrc/ck_*/` (e.g., `ck_gemm_a8w8/`, `ck_deepgemm/`) |
| CK tile interfaces | `csrc/py_itfs_ck/` |
| Raw CUDA/HIP kernels | `csrc/kernels/` |
| C++ interfaces | `csrc/cpp_itfs/` |
| Headers | `csrc/include/` |
| Pybind modules | `csrc/pybind/` |
| Python ops | `aiter/ops/*.py` (using `@compile_ops`) |
| Build config | `aiter/jit/optCompilerConfig.json` |
| Tuning configs | `aiter/configs/*.csv` |
| Tests | `op_tests/test_*.py`, `op_tests/cpp/` |

## CK Architecture Overview

```
Python Op (@compile_ops)
  → JIT loads module from optCompilerConfig.json
  → blob_gen_cmd runs (gen_instances.py for CK codegen)
  → C++ sources compiled with HIP/ROCm
  → Pybind exposes C++ function
  → Python op calls extension
```

## Writing a CK Kernel

### Step 1: Create Kernel Source

Place in `csrc/ck_yourkernel/`. Example GEMM:

```cpp
// csrc/ck_gemm_yourname/gemm_yourname.cu
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
#include "gemm_yourname_common.cuh"

// Template alias for the CK device operation
template <typename AType, typename BType, typename CType, typename ScaleType>
using DeviceGemmOp = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
    ck::tensor_layout::gemm::RowMajor,    // A layout
    ck::tensor_layout::gemm::ColumnMajor, // B layout  
    ck::Tuple<ScaleType, ScaleType>,       // D types (scales)
    CType,                                  // E type (output)
    AType, BType, float,                   // compute types
    // Tile sizes and pipeline config...
    256, 256, 128, 64,    // MPerBlock, NPerBlock, KPerBlock, AK1
    32, 32,                // MPerXdl, NPerXdl
    4, 2,                  // MRepeat, NRepeat
    // Thread cluster...
    4, 64, 1               // ABlockTransfer dims
>;

// Kernel dispatch logic
template <typename ABType, typename DType, typename EType>
bool rowwise_dispatch(int M, int N, int K,
                      const ABType* A, const ABType* B,
                      const DType* a_scale, const DType* b_scale,
                      EType* C, const EType* bias, int splitK)
{
    // 1. Check tuned lookup table (from CSV)
    auto it = RowwiseKernelMap<ABType, DType, EType>::map.find({M, N, K});
    if (it != RowwiseKernelMap<ABType, DType, EType>::map.end()) {
        return it->second(A, B, a_scale, b_scale, C, bias, M, N, K, splitK);
    }
    
    // 2. Heuristic dispatch by shape ranges
    return rowwise_heuristic_dispatch<ABType, DType, EType>(
        M, N, K, A, B, a_scale, b_scale, C, bias, splitK);
}

// Main entrypoint (called from pybind)
void gemm_yourname(
    torch::Tensor XQ,      // (M, K) quantized input
    torch::Tensor WQ,      // (N, K) quantized weight
    torch::Tensor x_scale, // (M, 1) scale
    torch::Tensor w_scale, // (1, N) scale
    torch::Tensor Y,       // (M, N) output
    torch::Tensor bias,    // optional (N,)
    int splitK)
{
    int M = XQ.size(0), K = XQ.size(1), N = WQ.size(0);
    
    // Dtype dispatch
    if (XQ.dtype() == torch::kInt8) {
        rowwise_dispatch<int8_t, float, ck::bhalf_t>(
            M, N, K, XQ.data_ptr<int8_t>(), WQ.data_ptr<int8_t>(),
            x_scale.data_ptr<float>(), w_scale.data_ptr<float>(),
            Y.data_ptr<ck::bhalf_t>(), bias.data_ptr<ck::bhalf_t>(), splitK);
    }
    // ... other dtype combinations
}
```

### Step 2: Create Instance Generator

`csrc/ck_gemm_yourname/gen_instances.py`:

```python
#!/usr/bin/env python3
"""Generate CK kernel instances for different tile configurations."""
import argparse
import os

TEMPLATE = '''
#include "gemm_yourname_common.cuh"

// Instance: {name}
using {name} = DeviceGemmOp<
    {a_type}, {b_type}, {c_type}, {scale_type},
    {m_per_block}, {n_per_block}, {k_per_block},
    {m_per_xdl}, {n_per_xdl}, {m_repeat}, {n_repeat}
>;

REGISTER_KERNEL({name}, {a_type}, {scale_type}, {c_type});
'''

CONFIGS = [
    {"name": "yourname_256x256x128", "m_per_block": 256, "n_per_block": 256, "k_per_block": 128, ...},
    {"name": "yourname_128x128x64",  "m_per_block": 128, "n_per_block": 128, "k_per_block": 64, ...},
    # ... more configs
]

def generate(working_path, tune_file=None):
    os.makedirs(working_path, exist_ok=True)
    for cfg in CONFIGS:
        filepath = os.path.join(working_path, f"{cfg['name']}.cu")
        with open(filepath, 'w') as f:
            f.write(TEMPLATE.format(**cfg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_path", required=True)
    parser.add_argument("--tune_file", default=None)
    args = parser.parse_args()
    generate(args.working_path, args.tune_file)
```

### Step 3: Create Header

`csrc/include/gemm_yourname.h`:

```cpp
#pragma once
#include <torch/extension.h>

void gemm_yourname(
    torch::Tensor XQ,
    torch::Tensor WQ,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    torch::Tensor Y,
    torch::Tensor bias,
    int splitK = 0);
```

### Step 4: Create Pybind Module

`csrc/pybind/gemm_yourname_pybind.cu`:

```cpp
#include "rocm_ops.hpp"
#include "gemm_yourname.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_yourname", &gemm_yourname, "GEMM yourname",
          py::arg("XQ"), py::arg("WQ"),
          py::arg("x_scale"), py::arg("w_scale"),
          py::arg("Y"), py::arg("bias"),
          py::arg("splitK") = 0);
}
```

Or use the macro pattern in `rocm_ops.hpp`:

```cpp
// In rocm_ops.hpp, add:
#define GEMM_YOURNAME_PYBIND \
    m.def("gemm_yourname", &gemm_yourname, "GEMM yourname", \
          py::arg("XQ"), py::arg("WQ"), \
          py::arg("x_scale"), py::arg("w_scale"), \
          py::arg("Y"), py::arg("bias"), \
          py::arg("splitK") = 0);

// In pybind file:
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    GEMM_YOURNAME_PYBIND;
}
```

### Step 5: Register in Build Config

Add to `aiter/jit/optCompilerConfig.json`:

```json
{
    "module_gemm_yourname": {
        "srcs": [
            "{AITER_CSRC_DIR}/pybind/gemm_yourname_pybind.cu",
            "{AITER_CSRC_DIR}/ck_gemm_yourname/gemm_yourname.cu"
        ],
        "extra_include": [
            "{AITER_CSRC_DIR}/ck_gemm_yourname/",
            "{CK_DIR}/include/"
        ],
        "flags_extra_cc": [],
        "flags_extra_hip": ["-O3"],
        "blob_gen_cmd": "{AITER_CSRC_DIR}/ck_gemm_yourname/gen_instances.py --working_path {} --tune_file {AITER_CONFIG_DIR}/yourname_tuned_gemm.csv"
    }
}
```

### Step 6: Create Python Op

`aiter/ops/gemm_op_yourname.py`:

```python
from aiter.jit.core import compile_ops
import torch

def gen_fake_tensors(XQ, WQ, x_scale, w_scale, Y, bias, splitK=0):
    """Fake tensors for torch.compile tracing."""
    M, K = XQ.shape
    N = WQ.shape[0]
    return torch.empty(M, N, dtype=torch.bfloat16, device=XQ.device)

@compile_ops("module_gemm_yourname", fc_name="gemm_yourname", gen_fake=gen_fake_tensors)
def gemm_yourname_ck(XQ, WQ, x_scale, w_scale, Y, bias, splitK=0):
    ...  # Body is replaced by JIT-compiled extension

def gemm_yourname(x, w, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    """Public API with config lookup."""
    M, K = x.shape
    N = w.shape[0]
    Y = torch.empty(M, N, dtype=dtype, device=x.device)
    if bias is None:
        bias = torch.empty(0, device=x.device)
    # Load tuned config for splitK
    config = get_config(M, N, K)
    splitK = config.get("splitK", 0) if config else 0
    gemm_yourname_ck(x, w, x_scale, w_scale, Y, bias, splitK)
    return Y
```

### Step 7: Export from `aiter/__init__.py`

```python
from .ops.gemm_op_yourname import *  # noqa: F403,E402
```

## CK Tuning Config CSV Format

`aiter/configs/yourname_tuned_gemm.csv`:

```csv
cu_num,M,N,K,q_dtype_w,kernelId,splitK,us,kernelName,tflops,bw,errRatio
256,1,1280,8192,torch.int8,34,0,8.75,yourname_256x16x64x512,0.12,59.1,0.0
256,32,1280,8192,torch.int8,12,0,5.23,yourname_128x128x64,2.35,120.5,0.0
```

## CK Kernel for MHA (CK Tile FMHA)

For attention kernels, CK uses the CK Tile API:

```cpp
// csrc/py_itfs_ck/mha_fwd_kernels.cu
#include "mha_fwd.h"
#include "mha_common.h"

mha_fwd_args get_ck_fmha_fwd_args(
    torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
    torch::Tensor& out, torch::Tensor& softmax_lse,
    /* ... */)
{
    mha_fwd_args args;
    args.q_ptr = q.data_ptr();
    args.k_ptr = k.data_ptr();
    args.v_ptr = v.data_ptr();
    args.o_ptr = out.data_ptr();
    // Extract strides from tensors
    args.stride_q = q.stride(2);  // head_dim stride
    args.nhead_stride_q = q.stride(1);  // head stride
    args.batch_stride_q = q.stride(0);  // batch stride
    // ... same for K, V, O ...
    args.seqlen_q = q.size(2);
    args.seqlen_k = k.size(2);
    args.hdim = q.size(3);
    return args;
}
```

The `blob_gen_cmd` for MHA uses CK Tile's generator:
```
{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd --receipt 2 --output_dir {}
```

## Writing CK Kernel Tests

CK kernel tests are in `op_tests/test_*.py` (root level):

```python
import torch
import pytest
import aiter  # Uses the compile_ops JIT system

def test_gemm_ck():
    M, N, K = 128, 1024, 512
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    w = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
    x_scale = torch.rand(M, 1, device="cuda")
    w_scale = torch.rand(1, N, device="cuda")
    Y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    aiter.gemm_a8w8(x, w, x_scale, w_scale, Y, torch.empty(0, device="cuda"))
    ref = (x.float() @ w.float().T) * (x_scale @ w_scale)
    torch.testing.assert_close(Y.float(), ref.float(), atol=0.1, rtol=0.1)
```

## Build System Details

- **JIT compilation:** `aiter.jit.core.compile_ops` loads module config and compiles on first use
- **blob_gen_cmd:** Runs before compilation to generate CK instances or HSA blobs
- **CK_DIR:** Points to Composable Kernel source (3rdparty or system install)
- **PREBUILD_KERNELS:** `1`=attention, `2`=GEMM, `3`=all (used in `setup.py`)

## Prerequisites

Before writing CK kernels, read these foundational skills:
- [HIP Kernel Programming](../hip-kernel-programming/SKILL.md) - HIP language, CK library, PyBind patterns
- [AMD GPU Architecture](../amd-gpu-architecture/SKILL.md) - CDNA3/4 hardware, MFMA instructions

## Additional Resources

For CK template parameter tuning, MoE 2-stage CK patterns, and CK Tile API details, see [reference.md](reference.md).
