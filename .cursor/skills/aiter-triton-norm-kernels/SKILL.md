---
name: aiter-triton-norm-kernels
description: Write Triton normalization kernels, tests, and benchmarks for the aiter project. Use when creating or modifying RMSNorm, LayerNorm, GroupNorm, fused add+norm, or normalization+quantization kernels. Covers forward and backward passes, blocked and row-wise implementations.
---

# Triton Normalization Kernels in Aiter

## Project Layout

| Component | Path |
|-----------|------|
| Kernel code | `aiter/ops/triton/_triton_kernels/normalization/` |
| Python wrapper | `aiter/ops/triton/normalization/` |
| Tests | `op_tests/triton_tests/normalization/` |
| Benchmarks | `op_tests/op_benchmarks/triton/bench_rmsnorm.py` |
| CK backend | `aiter/ops/rmsnorm.py` (CK path via `compile_ops`) |

### Kernel Files

| Kernel | File | Description |
|--------|------|-------------|
| RMSNorm | `rmsnorm.py` | Core RMSNorm + variants |
| Fused add + RMSNorm + pad | `fused_add_rmsnorm_pad.py` | Add residual + RMSNorm + padding |
| General norm | `norm.py` | LayerNorm and other norms |

## RMSNorm Kernel Variants

The `rmsnorm.py` file contains multiple kernel variants:

1. **`_rms_norm_kernel`** - Standard RMSNorm (row-wise or blocked)
2. **`_quant_rms_norm_kernel`** - RMSNorm + per-token FP8 quantization
3. **`_fused_add_rmsnorm_kernel`** - Residual add + RMSNorm (reads x+residual, writes both)
4. **`_quant_fused_add_rmsnorm_kernel`** - Add + RMSNorm + quantization
5. **`_rmsnorm_bwd_triton`** - Backward pass
6. **`_rmsnorm_kernel_large_m_small_n`** - Optimized for large batch, small hidden dim

## Writing a Normalization Kernel

### Basic RMSNorm Pattern

```python
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    X_ptr, Y_ptr, W_ptr, Rstd_ptr,
    stride_x_row, stride_y_row,
    n_rows, n_cols, epsilon,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    RMSNorm: y = x * w / sqrt(mean(x^2) + eps)
    
    Two modes:
    - USE_BLOCKED=False: Each program processes one complete row
    - USE_BLOCKED=True: Programs cooperate on rows (for large n_cols)
    """
    if not USE_BLOCKED:
        # Row-wise mode: one program per row
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return
        
        # Load entire row
        X_row_ptr = X_ptr + row_idx * stride_x_row
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Compute RMS
        x_sq = x * x
        var = tl.sum(x_sq) / n_cols
        rstd = 1.0 / tl.sqrt(var + epsilon)
        
        # Store rstd for backward
        tl.store(Rstd_ptr + row_idx, rstd)
        
        # Normalize and apply weight
        w = tl.load(W_ptr + cols, mask=mask)
        y = x * rstd * w
        
        # Store output
        Y_row_ptr = Y_ptr + row_idx * stride_y_row
        tl.store(Y_row_ptr + cols, y.to(Y_ptr.type.element_ty), mask=mask)
    else:
        # Blocked mode for large hidden dims
        # Programs iterate over multiple rows, each with multiple column blocks
        pass
```

### Fused Add + RMSNorm Pattern

Common in transformer residual connections:

```python
@triton.jit
def _fused_add_rmsnorm_kernel(
    X_ptr, Residual_ptr, Y_ptr, W_ptr, Rstd_ptr,
    stride_x_row, stride_r_row, stride_y_row,
    n_rows, n_cols, epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # Load x and residual
    x = tl.load(X_ptr + row_idx * stride_x_row + cols, mask=mask).to(tl.float32)
    residual = tl.load(Residual_ptr + row_idx * stride_r_row + cols, mask=mask).to(tl.float32)
    
    # Add residual
    x = x + residual
    
    # Write updated residual back (for next layer)
    tl.store(Residual_ptr + row_idx * stride_r_row + cols, x.to(Residual_ptr.type.element_ty), mask=mask)
    
    # RMSNorm
    var = tl.sum(x * x) / n_cols
    rstd = 1.0 / tl.sqrt(var + epsilon)
    w = tl.load(W_ptr + cols, mask=mask)
    y = x * rstd * w
    
    tl.store(Y_ptr + row_idx * stride_y_row + cols, y.to(Y_ptr.type.element_ty), mask=mask)
```

### Norm + Quantization Pattern

Fuse normalization with FP8 quantization:

```python
@triton.jit
def _per_token_quant(x, DTYPE_MAX: tl.constexpr, CLAMP_MAX: tl.constexpr):
    """Per-token dynamic quantization helper."""
    amax = tl.max(tl.abs(x))
    scale = amax / DTYPE_MAX
    scale = tl.where(scale == 0, 1.0, scale)
    x_quant = tl.clamp(x / scale, -CLAMP_MAX, CLAMP_MAX)
    return x_quant, scale

@triton.jit
def _quant_rms_norm_kernel(
    X_ptr, Y_ptr, Y_Scale_ptr, W_ptr, Rstd_ptr,
    ...,
    DTYPE_MAX: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
):
    # ... RMSNorm computation ...
    y = x * rstd * w
    # Quantize output
    y_quant, scale = _per_token_quant(y, DTYPE_MAX, CLAMP_MAX)
    tl.store(Y_ptr + ..., y_quant.to(tl.float8e4m3fn), mask=mask)
    tl.store(Y_Scale_ptr + row_idx, scale)
```

### Python Wrapper Pattern

```python
import torch
import triton

def rmsnorm(x, weight, epsilon=1e-6, out=None):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    USE_BLOCKED = n_cols > 8192  # Switch to blocked for large hidden dims
    
    if out is None:
        out = torch.empty_like(x)
    rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    
    if USE_BLOCKED:
        NUM_PRGMS = min(n_rows, 256)
        grid = (NUM_PRGMS,)
    else:
        grid = (n_rows,)
    
    _rms_norm_kernel[grid](
        x, out, weight, rstd,
        x.stride(0), out.stride(0),
        n_rows, n_cols, epsilon,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BLOCKED=USE_BLOCKED,
        NUM_PRGMS=NUM_PRGMS if USE_BLOCKED else 1,
    )
    return out, rstd
```

## Writing Normalization Tests

```python
import torch
import pytest

def torch_rmsnorm(x, weight, epsilon=1e-6):
    """Reference RMSNorm in float32."""
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_f32 * torch.rsqrt(var + epsilon)
    return (x_norm * weight.to(torch.float32)).to(x.dtype)

def generate_rmsnorm_inputs(M, N, dtype, device="cuda"):
    x = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    return x, weight

@pytest.mark.parametrize("M,N,dtype", [
    (M, N, dtype)
    for M in [1, 32, 128, 1024, 4096]
    for N in [512, 1024, 4096, 8192, 16384]
    for dtype in [torch.float16, torch.bfloat16, torch.float32]
])
def test_rmsnorm(M, N, dtype):
    x, weight = generate_rmsnorm_inputs(M, N, dtype)
    ref = torch_rmsnorm(x, weight)
    out, _ = rmsnorm(x, weight)
    # Tolerances: fp16/bf16 → atol=1e-2, fp32 → atol=1e-4
    atol = 1e-4 if dtype == torch.float32 else 1e-2
    rtol = atol
    torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

def test_fused_add_rmsnorm(M, N, dtype):
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    residual = torch.randn(M, N, dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")
    # Reference
    ref_sum = x + residual
    ref_out = torch_rmsnorm(ref_sum, weight)
    # Kernel (modifies residual in-place)
    residual_copy = residual.clone()
    out = fused_add_rmsnorm(x, residual_copy, weight)
    torch.testing.assert_close(ref_out, out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_sum.to(dtype), residual_copy, atol=1e-2, rtol=1e-2)
```

## Writing Normalization Benchmarks

Norm kernels are memory-bandwidth bound, so the primary metric is **bandwidth (GB/s)**:

```python
def bench_rmsnorm_fn(M, N, metric, dtype=torch.bfloat16):
    x, weight = generate_rmsnorm_inputs(M, N, dtype)
    # Memory: read x + weight, write out + rstd
    mem = (M * N * x.element_size()) * 2 + N * weight.element_size() + M * 4
    ms = triton.testing.do_bench(lambda: rmsnorm(x, weight), warmup=25, rep=100)
    if metric == "time": return ms
    elif metric == "bandwidth": return mem / (ms * 1e-3) * 1e-9  # GB/s
```

**bench_schema.yaml:**
```yaml
rmsnorm:
  input_columns: [model_name, M, N]
  output_columns: [Bandwidth_(GB/s)]
```

## Prerequisites

Before writing normalization kernels, read these foundational skills:
- [Triton Language Guide](../triton-language-guide/SKILL.md) - Triton API and AMD optimizations
- [GPU Kernel Algorithms](../gpu-kernel-algorithms/SKILL.md) - RMSNorm algorithm, roofline model

## Backward Pass

RMSNorm backward computes dx and dweight:

```python
@triton.jit
def _rmsnorm_bwd_triton(
    DY_ptr, X_ptr, W_ptr, Rstd_ptr,
    DX_ptr, DW_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # dx = (dy * w * rstd) - x * rstd^3 * mean(dy * w * x) / n_cols
    # dw = sum_over_rows(dy * x * rstd)
```

For dweight reduction across rows, use a two-pass approach:
1. `_rmsnorm_bwd_triton` - Compute dx and partial dw per row-block
2. `_rmsnorm_bwd_dg_reduce_triton` - Reduce partial dw across blocks
