---
name: aiter-triton-moe-kernels
description: Write Triton Mixture-of-Experts (MoE) kernels, tests, and benchmarks for the aiter project. Use when creating or modifying MoE GEMM, token routing, expert gating, top-k softmax, align block size, or fused MoE kernels. Covers both single-stage and two-stage MoE patterns.
---

# Triton MoE Kernels in Aiter

## Project Layout

| Component | Path |
|-----------|------|
| Kernel code | `aiter/ops/triton/_triton_kernels/moe/` |
| Python wrapper | `aiter/ops/triton/moe/` |
| Tests | `op_tests/triton_tests/moe/` |
| Benchmarks | `op_tests/op_benchmarks/triton/bench_moe*.py` |
| High-level API | `aiter/fused_moe.py` |
| Configs | `aiter/configs/tuned_fmoe.csv`, `aiter/ops/triton/configs/moe/` |

### MoE Kernel Files

| Kernel | File | Description |
|--------|------|-------------|
| MoE GEMM | `moe_op.py` | Core MoE grouped GEMM |
| MoE E2E | `moe_op_e2e.py` | End-to-end MoE (routing + GEMM) |
| MoE SiLU fused | `moe_op_silu_fused.py` | MoE GEMM with fused SiLU activation |
| MoE A8W8 | `moe_op_gemm_a8w8.py` | INT8 quantized MoE GEMM |
| Align block size | `moe_align_block_size.py` | Token-to-block alignment |
| Quant MoE | `quant_moe.py` | MoE with quantization |
| Routing | `moe_routing_sigmoid_top1_fused.py` | Fused sigmoid routing + top-1 |

## MoE Architecture Overview

MoE kernels handle the unique challenge of dispatching tokens to different experts:

```
Input tokens → Router (top-k gating) → Sort by expert → Grouped GEMM → Scatter back
```

### Key Concepts

1. **Token routing:** Each token is assigned to top-k experts via softmax/sigmoid gating
2. **Sorting:** Tokens are reordered by expert for efficient batched processing
3. **Grouped GEMM:** Single kernel processes all experts, each with different token counts
4. **Two-stage:** Stage 1 (gate+up GEMM) → activation → Stage 2 (down GEMM)

## Writing a MoE Kernel

### Core MoE GEMM Pattern

```python
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd, pid_grid

_moe_gemm_repr = make_kernel_repr("_moe_gemm_kernel", [
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
    "GROUP_SIZE_M", "EVEN_K", "NUM_XCDS",
])

@triton.heuristics({
    "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
})
@triton.jit(repr=_moe_gemm_repr)
def _moe_gemm_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Token-expert mapping
    sorted_token_ids_ptr,   # Sorted indices of tokens
    expert_ids_ptr,         # Expert ID for each token group
    num_tokens_post_padded_ptr,  # Total tokens after padding
    topk_weights_ptr,       # Gating weights per token-expert pair
    # Scales (for quantized)
    a_scale_ptr, b_scale_ptr,
    # Dimensions
    N, K,
    EM,  # Total padded tokens across all experts
    num_valid_tokens,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,  # b has expert dimension
    stride_cm, stride_cn,
    # Expert count
    top_k,
    # Meta-parameters
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    # Grid: each program handles one (M-block, N-block) tile
    pid = tl.program_id(axis=0)
    # ... XCD remapping, grouped ordering ...
    
    # Determine which expert this tile belongs to
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid * BLOCK_SIZE_M >= num_tokens_post_padded:
        return  # Early exit for padding tiles
    
    # Load sorted token IDs for this M-block
    offs_token = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    sorted_token_ids = tl.load(sorted_token_ids_ptr + offs_token)
    token_mask = sorted_token_ids < num_valid_tokens
    
    # Expert for this block
    expert_id = tl.load(expert_ids_ptr + pid_m)
    
    # Offset B (weight) by expert
    b_ptr += expert_id * stride_be
    
    # Standard GEMM loop with gather for A
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + sorted_token_ids[:, None] * stride_am + offs_k[None, :] * stride_ak
    # ... K-loop, accumulate, apply scales ...
    
    # Apply routing weight
    if MUL_ROUTED_WEIGHT:
        weights = tl.load(topk_weights_ptr + sorted_token_ids)
        accumulator *= weights[:, None]
    
    # Store with gather pattern
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + sorted_token_ids[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=token_mask[:, None] & (offs_cn[None, :] < N))
```

### Key MoE Patterns

**Expert-indexed weight:** `b_ptr + expert_id * stride_be` (weights are `[num_experts, K, N]`)

**Sorted token gather:** Input A is indexed via `sorted_token_ids` not sequential

**Routing weight multiplication:** `MUL_ROUTED_WEIGHT` flag for applying gating weights

**Align block size:** Pads token counts per expert to `BLOCK_SIZE_M` multiples:

```python
@triton.jit
def _moe_align_block_size_kernel(
    topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
    num_experts, block_size, numel,
):
    # Histogram of tokens per expert
    # Pad each expert's count to block_size multiple
    # Write sorted indices and expert IDs
```

## Writing MoE Tests

```python
import torch
import pytest

def torch_moe_ref(x, w, sorted_token_ids, expert_ids, topk_weights, top_k):
    """Reference MoE GEMM in PyTorch."""
    out = torch.zeros(x.shape[0] * top_k, w.shape[-1], dtype=x.dtype, device=x.device)
    for i, expert_id in enumerate(expert_ids):
        # Get tokens for this expert block
        token_ids = sorted_token_ids[i * block_size:(i+1) * block_size]
        valid = token_ids < x.shape[0] * top_k
        tokens = x[token_ids[valid] // top_k]
        expert_w = w[expert_id]
        result = tokens @ expert_w.T
        out[token_ids[valid]] = result * topk_weights[token_ids[valid], None]
    return out

def input_helper(M, N, K, E, top_k, dtype):
    """Generate MoE test inputs."""
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    w = torch.randn(E, N, K, dtype=dtype, device="cuda")  # [experts, N, K]
    scores = torch.randn(M, E, device="cuda")
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    # Sort and align
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, E)
    return x, w, sorted_token_ids, expert_ids, topk_weights, num_tokens_post_padded

@pytest.mark.parametrize("M,N,K,E,top_k", [
    (32, 1024, 512, 8, 2),
    (128, 4096, 2048, 64, 2),
    (1, 1280, 8192, 8, 1),
])
def test_moe_gemm(M, N, K, E, top_k):
    x, w, sorted_ids, expert_ids, weights, ntp = input_helper(M, N, K, E, top_k, torch.bfloat16)
    ref = torch_moe_ref(x, w, sorted_ids, expert_ids, weights, top_k)
    out = moe_gemm(x, w, sorted_ids, expert_ids, weights, ntp, top_k)
    torch.testing.assert_close(ref, out, atol=1e-1, rtol=1e-1)
```

**Tolerances:** MoE typically uses `atol=1e-1, rtol=1e-1` (looser due to expert routing and gather patterns). INT4 variants use `atol=2e-1`.

## Writing MoE Benchmarks

```python
def bench_moe_fn(M, N, K, E, top_k, metric, impl):
    x, w, sorted_ids, expert_ids, weights, ntp = input_helper(M, N, K, E, top_k, torch.bfloat16)
    flops = 2.0 * M * top_k * N * K  # Each token does top_k expert GEMMs
    mem = (M * K + E * N * K + M * top_k * N) * 2  # BF16
    ms = triton.testing.do_bench(
        lambda: impl(x, w, sorted_ids, expert_ids, weights, ntp, top_k),
        warmup=25, rep=100)
    if metric == "time": return ms
    elif metric == "throughput": return flops / ms * 1e-9
    elif metric == "bandwidth": return mem / (ms * 1e-3) * 1e-9
```

**bench_schema.yaml:**
```yaml
moe:
  input_columns: [model, M, N, K, E, top_k]
  output_columns: [Time_(ms), TFLOPS, Bandwidth_(GB/s)]
```

## Two-Stage MoE Pattern

The high-level `fused_moe()` in `aiter/fused_moe.py` orchestrates:

1. `moe_sorting_fwd()` - Sort tokens by expert
2. Stage 1: `fmoe_g1u1()` or Triton MoE GEMM (gate + up projection)
3. Activation (SiLU gate)
4. Stage 2: MoE GEMM (down projection)

Configs loaded from `aiter/configs/tuned_fmoe.csv` keyed by `(cu_num, tokens, model_dim, inter_dim, experts, topk, activation, dtype)`.

## Prerequisites

Before writing MoE kernels, read these foundational skills:
- [Triton Language Guide](../triton-language-guide/SKILL.md) - Triton API and AMD optimizations
- [GPU Kernel Algorithms](../gpu-kernel-algorithms/SKILL.md) - MoE dispatch, GEMM tiling, quantization

## Additional Resources

For detailed sorting algorithms, INT4/INT8 MoE quantization patterns, and CK/ASM MoE backends, see [reference.md](reference.md).
