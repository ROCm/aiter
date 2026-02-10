---
name: aiter-triton-quant-kernels
description: Write Triton quantization kernels, tests, and benchmarks for the aiter project. Use when creating or modifying FP8 quantization, INT8 quantization, MXFP4 quantization, per-tensor/per-token/per-block/per-group scaling, or fused quantization+normalization kernels.
---

# Triton Quantization Kernels in Aiter

## Project Layout

| Component | Path |
|-----------|------|
| Kernel code | `aiter/ops/triton/_triton_kernels/quant/` |
| Python wrapper | `aiter/ops/triton/quant/` |
| Tests | `op_tests/triton_tests/quant/` |
| CK backend | `aiter/ops/quant.py` |

### Kernel Files

| Kernel | File | Description |
|--------|------|-------------|
| Basic quant | `quant.py` | Per-tensor, per-token FP8/I8, MXFP4 |
| Fused FP8 | `fused_fp8_quant.py` | RMSNorm + FP8, flatten + FP8 group quant |
| Fused MXFP4 | `fused_mxfp4_quant.py` | RMSNorm + MXFP4, dynamic MoE sort + MXFP4 |

### Quantization Types

| Type | Scale Granularity | Use Case |
|------|------------------|----------|
| Per-tensor static | Single scalar | Post-training quantization |
| Per-tensor dynamic | Single scalar (computed at runtime) | Dynamic range |
| Per-token | One scale per row (M,) | Activation quantization |
| Per-block (blockscale) | 2D scale (M/bm, K/bk) | Block-wise GEMM |
| Per-group (1x128) | Scale per 128 elements | Fine-grained quant |
| MXFP4 (1x32) | Scale per 32 elements | Microscaling FP4 |

## Writing a Quantization Kernel

### Per-Tensor Static Quantization

```python
import triton
import triton.language as tl

@triton.jit
def _static_per_tensor_quant_kernel(
    X_ptr, Y_ptr,
    scale,  # Pre-computed scale factor
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(X_ptr + offs, mask=mask).to(tl.float32)
    y = tl.clamp(x / scale, -DTYPE_MAX, DTYPE_MAX)
    tl.store(Y_ptr + offs, y.to(Y_ptr.type.element_ty), mask=mask)
```

### Per-Token Dynamic Quantization

```python
@triton.jit
def _dynamic_per_token_quant_kernel(
    X_ptr, Y_ptr, Scale_ptr,
    n_rows, n_cols,
    stride_x_row, stride_y_row,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # Load row
    x = tl.load(X_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute per-token scale
    amax = tl.max(tl.abs(x))
    scale = amax / DTYPE_MAX
    scale = tl.where(scale == 0.0, 1.0, scale)
    
    # Quantize
    y = tl.clamp(x / scale, -CLAMP_MAX, CLAMP_MAX)
    
    # Store
    tl.store(Y_ptr + row_idx * stride_y_row + cols, y.to(Y_ptr.type.element_ty), mask=mask)
    tl.store(Scale_ptr + row_idx, scale)
```

### MXFP4 Quantization Pattern

MXFP4 uses a shared exponent per 32 elements with 4-bit mantissa:

```python
@triton.jit
def _mxfp4_quant_op(val):
    """Convert float to MXFP4 format via bit manipulation."""
    # Extract sign, exponent, mantissa from float
    val_u32 = val.to(tl.uint32, bitcast=True)
    sign = (val_u32 >> 31) & 1
    exp = (val_u32 >> 23) & 0xFF
    mantissa = val_u32 & 0x7FFFFF
    # Truncate mantissa to 2 bits (4-bit total with sign)
    # Apply shared exponent scaling
    # Pack two FP4 values into one byte
    return packed_val

@triton.heuristics({"EVEN_M_N": lambda args: args["M"] * args["N"] % args["BLOCK_SIZE"] == 0})
@triton.jit
def _dynamic_mxfp4_quant_kernel(
    X_ptr, Y_ptr, Scale_ptr,
    M, N,
    stride_x_row,
    BLOCK_SIZE: tl.constexpr,  # Elements per program (typically 32)
    GROUP_SIZE: tl.constexpr,  # Scale group size (32 for MXFP4)
    EVEN_M_N: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles GROUP_SIZE elements
    offs = pid * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    
    x = tl.load(X_ptr + offs, mask=offs < M * N).to(tl.float32)
    
    # Shared exponent = max exponent in group
    amax = tl.max(tl.abs(x))
    shared_exp = tl.floor(tl.log2(amax)) if amax > 0 else 0
    scale = tl.exp2(shared_exp)
    
    # Quantize with shared exponent
    x_scaled = x / scale
    x_quant = _mxfp4_quant_op(x_scaled)
    
    tl.store(Scale_ptr + pid, scale)
    # Pack and store MXFP4 values (2 per byte)
```

### Fused RMSNorm + FP8 Quantization

```python
@triton.jit
def _fused_rms_fp8_quant_kernel(
    X_ptr, W_ptr, Y_ptr, Scale_ptr,
    n_rows, n_cols, epsilon,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # RMSNorm
    x = tl.load(X_ptr + row_idx * n_cols + cols, mask=mask).to(tl.float32)
    var = tl.sum(x * x) / n_cols
    rstd = 1.0 / tl.sqrt(var + epsilon)
    w = tl.load(W_ptr + cols, mask=mask).to(tl.float32)
    y = x * rstd * w
    
    # Per-token FP8 quantization
    amax = tl.max(tl.abs(y))
    scale = amax / DTYPE_MAX
    scale = tl.where(scale == 0.0, 1.0, scale)
    y_quant = tl.clamp(y / scale, -DTYPE_MAX, DTYPE_MAX)
    
    tl.store(Y_ptr + row_idx * n_cols + cols, y_quant.to(tl.float8e4m3fn), mask=mask)
    tl.store(Scale_ptr + row_idx, scale)
```

## Python Wrapper

```python
import torch
from aiter.ops.triton.utils.types import get_fp8_dtypes

def dynamic_per_token_quant_fp8(x):
    """Quantize activation tensor to FP8 with per-token scales."""
    _, e4m3_type = get_fp8_dtypes()
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x, dtype=e4m3_type)
    scales = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = (M,)
    _dynamic_per_token_quant_kernel[grid](
        x, y, scales, M, N, x.stride(0), y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE_MAX=torch.finfo(e4m3_type).max,
        CLAMP_MAX=torch.finfo(e4m3_type).max,
    )
    return y, scales
```

## Writing Quantization Tests

```python
import torch
import pytest
from aiter.ops.triton.utils.types import get_fp8_dtypes

def torch_dynamic_per_token_quant(x, dtype_max):
    """Reference per-token quantization."""
    x_f32 = x.to(torch.float32)
    amax = x_f32.abs().amax(dim=-1, keepdim=True)
    scale = amax / dtype_max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (x_f32 / scale).clamp(-dtype_max, dtype_max), scale.squeeze(-1)

@pytest.mark.parametrize("M,N,dtype", [
    (M, N, dtype)
    for M in [1, 32, 128, 1024]
    for N in [512, 1024, 4096, 8192]
    for dtype in [torch.float16, torch.bfloat16]
])
def test_per_token_quant(M, N, dtype):
    _, e4m3 = get_fp8_dtypes()
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    ref_y, ref_scale = torch_dynamic_per_token_quant(x, torch.finfo(e4m3).max)
    out_y, out_scale = dynamic_per_token_quant_fp8(x)
    torch.testing.assert_close(ref_scale, out_scale, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_y.to(e4m3), out_y, atol=1e-1, rtol=1e-1)
```

**Tolerances:**
- Static quantization: `atol=1e-2, rtol=1e-2`
- Dynamic quantization: `atol=1e-1, rtol=1e-1`
- MXFP4: Looser due to 4-bit precision

## Quantization Exports

`aiter/ops/triton/quant/__init__.py` exports:
- `static_per_tensor_quant_fp8_i8`
- `dynamic_per_tensor_quant_fp8_i8`
- `dynamic_per_token_quant_fp8_i8`
- `dynamic_mxfp4_quant`
- `fused_rms_fp8_per_tensor_static_quant`
- `fused_rms_fp8_group_quant`
- `fused_flatten_fp8_group_quant`
- `fused_rms_mxfp4_quant`
- `fused_flatten_mxfp4_quant`

## Prerequisites

Before writing quantization kernels, read these foundational skills:
- [Triton Language Guide](../triton-language-guide/SKILL.md) - Triton API, data types, AMD optimizations
- [GPU Kernel Algorithms](../gpu-kernel-algorithms/SKILL.md) - Quantization schemes (per-tensor, per-token, MXFP4)

## Key Design Patterns

- **DTYPE_MAX/CLAMP_MAX as constexpr:** Avoids runtime branching for dtype limits
- **Scale = 1.0 when amax = 0:** Prevents division by zero
- **float32 accumulation:** All quant math in f32 for accuracy
- **Atomic max for per-tensor dynamic:** `tl.atomic_max` when multiple blocks compute global scale
- **Pack 2 MXFP4 values per byte:** Output tensor has half the elements
