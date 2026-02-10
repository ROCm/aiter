---
name: aiter-triton-attention-kernels
description: Write Triton attention kernels, tests, and benchmarks for the aiter project. Use when creating or modifying MHA (Multi-Head Attention), MLA (Multi-head Latent Attention), PA (Paged Attention) decode/prefill kernels, flash attention, or any attention-related Triton kernel.
---

# Triton Attention Kernels in Aiter

## Project Layout

| Component | Path |
|-----------|------|
| Kernel code | `aiter/ops/triton/_triton_kernels/attention/` |
| Python wrapper | `aiter/ops/triton/attention/` |
| Flash attention (AMD) | `aiter/ops/triton/_triton_kernels/attention/flash_attn_triton_amd/` |
| Tests | `op_tests/triton_tests/attention/` |
| Benchmarks | `op_tests/op_benchmarks/triton/bench_mha.py`, `bench_pa_decode.py`, `bench_pa_prefill.py` |
| Configs | `aiter/ops/triton/configs/{arch}-MHA-DEFAULT.json` |
| High-level API | `aiter/mla.py`, `aiter/paged_attn.py` |

### Attention Kernel Types

| Type | File | Description |
|------|------|-------------|
| MHA Forward | `attention/mha.py` | Multi-head attention (flash attention style) |
| PA Decode | `attention/pa_decode.py` | Paged attention for decoding (single query) |
| PA Prefill | `attention/pa_prefill.py` | Paged attention for prefilling (multi-query) |
| FAv3 SAGE | `attention/fav3_sage_attention.py` | Flash Attention v3 with SAGE |
| Flash Attn AMD | `flash_attn_triton_amd/` | AMD-optimized flash attention (fwd/bwd) |

## Writing an Attention Kernel

### Kernel Structure Pattern

Attention kernels are structured as **helper functions + main kernel**:

```python
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Helper JIT functions
@triton.jit
def _attn_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m, seqlen_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Inner attention loop: QK matmul, mask, softmax, V accumulation."""
    lo = 0
    hi = seqlen_k if not IS_CAUSAL else tl.minimum(seqlen_k, start_m + BLOCK_M)
    
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        # QK^T
        qk = tl.dot(q, tl.trans(k))
        # Causal mask
        if IS_CAUSAL:
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk * 1.44269504)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        # PV
        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i

_attn_fwd_repr = make_kernel_repr("_attn_fwd", [
    "IS_CAUSAL", "BLOCK_M", "BLOCK_N", "BLOCK_DMODEL",
    "USE_ALIBI", "ENABLE_DROPOUT",
])

@triton.jit(repr=_attn_fwd_repr)
def _attn_fwd(
    Q, K, V, sm_scale, Out, LSE,
    # Strides: q, k, v, out (batch, head, seq, dim)
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # Dimensions
    nheads_q, nheads_k, seqlen_q, seqlen_k, head_dim,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    USE_ALIBI: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // nheads_q
    pid_h = pid_bh % nheads_q
    # GQA: map query head to KV head
    pid_kh = pid_h // (nheads_q // nheads_k)
    # ... offset pointers, call _attn_inner, normalize, store ...
```

### Key Attention Patterns

**Grid:** `(triton.cdiv(seqlen_q, BLOCK_M), batch * nheads_q)`

**GQA (Grouped Query Attention):**
```python
pid_kh = pid_h // (nheads_q // nheads_k)
```

**Online Softmax (log2-based for numerical stability):**
```python
m_ij = tl.maximum(m_i, tl.max(qk, 1))
qk -= m_ij[:, None]
p = tl.math.exp2(qk * 1.44269504)  # log2(e) = 1.44269504
```

**Variable-length sequences:** Use `cu_seqlens_q/k` (cumulative sequence lengths) arrays and `max_seqlen_q/k` for batch-aware indexing.

### Step 2: Python Wrapper

```python
import torch
import triton
from aiter.ops.triton._triton_kernels.attention.your_attn import _attn_fwd, _get_config

def attention_fwd(q, k, v, sm_scale=None, causal=False, out=None):
    B, H_Q, S_Q, D = q.shape
    _, H_K, S_K, _ = k.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    if out is None:
        out = torch.empty_like(q)
    lse = torch.empty((B, H_Q, S_Q), device=q.device, dtype=torch.float32)
    config = _get_config(q.dtype, causal)
    grid = lambda META: (triton.cdiv(S_Q, META["BLOCK_M"]), B * H_Q)
    _attn_fwd[grid](
        q, k, v, sm_scale, out, lse,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        H_Q, H_K, S_Q, S_K, D,
        causal, **config,
    )
    return out, lse
```

### Step 3: Config File

`aiter/ops/triton/configs/{arch}-MHA-DEFAULT.json`:

```json
{
  "fwd": {
    "default": {
      "BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 2,
      "num_warps": 4, "num_stages": 1, "matrix_instr_nonkdim": 16,
      "pre_load_v": true
    },
    "dropout_or_fp32": {
      "BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 1,
      "num_warps": 4, "num_stages": 1, "pre_load_v": false
    }
  }
}
```

## Writing Attention Tests

```python
import torch
import pytest
from aiter.test_mha_common import attention_ref  # Shared reference

def test_mha_fwd(batch, nheads_q, nheads_k, seqlen_q, seqlen_k, head_dim, causal, dtype):
    q = torch.randn(batch, nheads_q, seqlen_q, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, nheads_k, seqlen_k, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch, nheads_k, seqlen_k, head_dim, dtype=dtype, device="cuda")
    ref = attention_ref(q, k, v, causal=causal)
    out, _ = your_attention_fwd(q, k, v, causal=causal)
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
```

**Tolerances by scenario:**
- Standard MHA: `atol=1e-2, rtol=1e-2`
- FP8 attention: Use custom `fp8_assert_close` allowing 0.5% element failures at `atol=3e-1, rtol=2.5e-1`
- Variable-length: `atol=1e-1, rtol=1e-1`

## Writing Attention Benchmarks

```python
@triton.testing.perf_report([benchmark])
def bench_mha(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, metric, **kwargs):
    q = torch.randn(BATCH, HQ, N_CTX_Q, HEAD_DIM, dtype=dtype, device="cuda")
    k = torch.randn(BATCH, HK, N_CTX_K, HEAD_DIM, dtype=dtype, device="cuda")
    v = torch.randn(BATCH, HK, N_CTX_K, HEAD_DIM, dtype=dtype, device="cuda")
    flops = 2 * BATCH * HQ * N_CTX_Q * N_CTX_K * HEAD_DIM * 2  # QK + PV
    ms = triton.testing.do_bench(lambda: attn_fwd(q, k, v), warmup=25, rep=100)
    return flops / ms * 1e-9  # TFLOPS
```

**bench_schema.yaml entry:**
```yaml
mha:
  input_columns: [BATCH, HQ, HK, N_CTX_Q, N_CTX_K]
  output_columns: [fwd(TFLOPS)]
```

## Prerequisites

Before writing attention kernels, read these foundational skills:
- [Triton Language Guide](../triton-language-guide/SKILL.md) - Triton API and AMD optimizations
- [AMD GPU Architecture](../amd-gpu-architecture/SKILL.md) - CDNA3/4 hardware, memory hierarchy
- [GPU Kernel Algorithms](../gpu-kernel-algorithms/SKILL.md) - FlashAttention algorithm, online softmax, tiling

## Additional Resources

For paged attention decode/prefill patterns, flash attention AMD backend, and backward pass kernel patterns, see [reference.md](reference.md).
