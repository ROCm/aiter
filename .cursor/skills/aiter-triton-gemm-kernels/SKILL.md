---
name: aiter-triton-gemm-kernels
description: Write Triton GEMM (matrix multiplication) kernels, tests, and benchmarks for the aiter project. Use when the user wants to create, modify, or understand GEMM kernels including basic, batched, blockscale, fused, or feed-forward variants. Covers A8W8, A16W16, A4W4, AFP4WFP4 quantized GEMM patterns.
---

# Triton GEMM Kernels in Aiter

## Project Layout

| Component | Path |
|-----------|------|
| Kernel code | `aiter/ops/triton/_triton_kernels/gemm/` |
| Python wrapper | `aiter/ops/triton/gemm/` |
| Tests | `op_tests/triton_tests/gemm/` |
| Benchmarks | `op_tests/op_benchmarks/triton/bench_gemm_*.py` |
| Configs (JSON) | `aiter/ops/triton/configs/gemm/` |
| Bench schema | `op_tests/op_benchmarks/triton/bench_schema.yaml` |

### Subdirectory Structure

```
gemm/
  basic/      - gemm_a8w8, gemm_a16w16, gemm_a8w8_blockscale, gemm_afp4wfp4
  batched/    - batched_gemm_bf16, batched_gemm_a8w8, batched_gemm_afp4wfp4
  feed_forward/ - ff_a16w16, ff_a16w16_fused_gated, ff_a16w16_fused_ungated
  fused/      - fused_gemm_*_split_cat, fused_gemm_*_mul_add
```

## Writing a Triton GEMM Kernel

### Step 1: Create the Kernel File

Place in `aiter/ops/triton/_triton_kernels/gemm/{subcategory}/`. Follow this template:

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

# Kernel representation for caching
_gemm_yourname_repr = make_kernel_repr(
    "_gemm_yourname_kernel",
    ["HAS_BIAS", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
     "GROUP_SIZE_M", "EVEN_K", "GRID_MN", "NUM_XCDS"],
)

@triton.heuristics({
    "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    "GRID_MN": lambda args: (
        triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"])
    ),
})
@triton.jit(repr=_gemm_yourname_repr)
def _gemm_yourname_kernel(
    # Pointers (always first)
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, bias_ptr, c_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    # Meta-parameters (tl.constexpr)
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    # Stride assumptions for compiler optimization
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)

    # PID mapping with XCD remapping for AMD GPUs
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # ... XCD remapping logic (see reference.md) ...
    # ... grouped ordering for L2 cache reuse ...

    # Compute block pointers
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Main GEMM loop
    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scales, bias, store result
    # ... (see reference.md for full pattern) ...

def _get_config(M, N, K):
    return get_gemm_config("GEMM-YOURNAME", M, N, K)
```

### Step 2: Create the Python Wrapper

Place in `aiter/ops/triton/gemm/{subcategory}/`. Mirrors kernel location:

```python
from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.{subcategory}.{name} import _gemm_kernel, _get_config
from aiter.ops.triton.utils.device_info import get_num_xcds

def gemm_yourname(
    x: torch.Tensor, w: torch.Tensor,
    x_scale: torch.Tensor, w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    M, K = x.shape
    N, K = w.shape
    w = w.T  # Kernel expects (K, N)
    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    if config is None:
        config, _ = _get_config(M, N, K)
    grid = (triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    _gemm_kernel[grid](
        x, w, x_scale, w_scale, bias, y,
        M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
        bias is not None, NUM_XCDS=get_num_xcds(), **config,
    )
    return y
```

### Step 3: Register in Backward Compat Map

Add entry in `aiter/ops/triton/__init__.py`:

```python
_BACKWARD_COMPAT_MAP = {
    ...
    "gemm_yourname": "gemm.{subcategory}.gemm_yourname",
}
```

### Step 4: Add Autotuning Config

Create `aiter/ops/triton/configs/gemm/{arch}-GEMM-YOURNAME.json`:

```json
{
  "any": {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16
  }
}
```

Config supports M-based selection: `"M_LEQ_128"`, `"M_GEQ_256"`, `"any"` (fallback).

## Writing a GEMM Test

Place in `op_tests/triton_tests/gemm/{subcategory}/test_gemm_yourname.py`:

```python
import torch
import pytest
import torch.nn.functional as F
from aiter.ops.triton.gemm.{subcategory}.gemm_yourname import gemm_yourname
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype

def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    """Reference implementation in float32."""
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)

def generate_gemm_inputs(M, N, K, in_dtype, out_dtype, layout="TN", output=False):
    """Reusable input generator (also used by benchmarks)."""
    e5m2_type, e4m3_type = get_fp8_dtypes()
    dtype_max = {d: (torch.finfo(d) if d.is_floating_point else torch.iinfo(d)).max
                 for d in [e5m2_type, e4m3_type, torch.int8]}
    # Create x (M,K) and weight (N,K) with proper layout
    x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    weight = torch.randn((N, K), dtype=torch.float32, device="cuda")
    # Compute scales and quantize
    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / dtype_max[in_dtype]
    x = (x / x_scale).to(in_dtype)
    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / dtype_max[in_dtype]
    weight = (weight / w_scale.T).to(in_dtype)
    bias = torch.rand([1, N], dtype=torch.float32, device="cuda") * 10
    y = torch.empty((M, N), dtype=out_dtype, device="cuda") if output else None
    return x, weight, x_scale, w_scale, bias, y

def get_x_vals():
    """Standard test shapes."""
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(1, 1280, 8192), (32, 1280, 8192), (1, 1, 1)]
    return x_vals

@pytest.mark.parametrize("in_dtype, out_dtype, m, n, k", [
    (in_dtype, out_dtype, *shape)
    for in_dtype in ["fp8e4m3", "fp8e5m2", "int8"]
    for out_dtype in ["bf16", "fp16", "fp32"]
    for shape in get_x_vals()
])
def test_gemm(in_dtype, out_dtype, m, n, k):
    torch.cuda.empty_cache()
    in_dtype = str_to_torch_dtype[in_dtype]
    out_dtype = str_to_torch_dtype[out_dtype]
    x, weight, x_scale, w_scale, bias, y = generate_gemm_inputs(m, n, k, in_dtype, out_dtype)
    ref = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    result = gemm_yourname(x, weight, x_scale, w_scale, bias, out_dtype, y)
    # Tolerances: float→atol=0.02,rtol=1e-2; int→atol=1,rtol=1e-2
    torch.testing.assert_close(ref, result, atol=0.02, rtol=1e-2)
```

**Key test patterns:**
- Use `torch.cuda.empty_cache()` at test start for large tests
- Reference in float32, then cast to output dtype
- Use `pytest.skip()` for unsupported dtype/arch combos
- Input generator is shared with benchmarks

## Writing a GEMM Benchmark

Place in `op_tests/op_benchmarks/triton/bench_gemm_yourname.py`:

```python
import sys
import triton
import math
from aiter.ops.triton.gemm.{subcategory}.gemm_yourname import gemm_yourname
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.triton_tests.gemm.{subcategory}.test_gemm_yourname import generate_gemm_inputs
from op_tests.op_benchmarks.triton.utils.argparse import get_parser, add_argparse_ff, get_ff_args
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object, get_shape_benchmark_object,
    print_vgpr, get_caller_name_no_ext,
)

def bench_gemm_fn(M, N, K, metric, impl):
    c_dtype = str_to_torch_dtype["bf16"]
    x, weight, x_scale, w_scale, bias, y = generate_gemm_inputs(
        M, N, K, str_to_torch_dtype["fp8e4m3"], c_dtype, output=True)
    flops = 2.0 * M * N * K
    mem = (M*K)*x.element_size() + (N*K)*weight.element_size() + (M*N)*y.element_size()
    ms = triton.testing.do_bench(
        lambda: impl(x, weight, x_scale, w_scale, bias, c_dtype, y),
        warmup=25, rep=100)
    if metric == "time": return ms
    elif metric == "throughput": return flops / ms * 1e-9  # TFLOPS
    elif metric == "bandwidth": return mem / (ms * 1e-3) * 1e-9  # GB/s

def run_shape_benchmark(args, impl):
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)
    @triton.testing.perf_report([benchmark])
    def bench(M, N, K, metric, **kwargs):
        return bench_gemm_fn(M, N, K, metric, impl)
    bench.run(save_path="." if args.o else None, print_data=True)

def main():
    parser = get_parser(kernel_name="Your GEMM")
    parser = add_argparse_ff(parser)
    args, defaults = get_ff_args(parser)
    run_shape_benchmark(args, gemm_yourname)

if __name__ == "__main__":
    sys.exit(main())
```

**Register in bench_schema.yaml:**

```yaml
gemm_yourname:
  input_columns: [M, N, K]
  output_columns: [TFLOPS]
```

## Prerequisites

Before writing GEMM kernels, read these foundational skills:
- [Triton Language Guide](../triton-language-guide/SKILL.md) - Triton API and AMD optimizations
- [AMD GPU Architecture](../amd-gpu-architecture/SKILL.md) - CDNA3/4 hardware, MFMA, memory hierarchy
- [GPU Kernel Algorithms](../gpu-kernel-algorithms/SKILL.md) - GEMM tiling, Split-K, roofline model

## Additional Resources

For detailed XCD remapping logic, config format specs, and batched/fused GEMM patterns, see [reference.md](reference.md).
