# AITER Attention Variants & Backend Guide

This guide documents all attention variants available in AITER, their backend support, and how to choose the right one for your use case.

---

## Quick Reference: Which Attention Should I Use?

| Use Case | Recommended Variant | Backend | Why |
|----------|-------------------|---------|-----|
| **Standard training/inference** | MHA (Flash Attention) | CK (C++) | Mature, supports fwd+bwd, broadest dtype support |
| **LLM inference (decode)** | Paged Attention Decode | ASM | Best latency, memory-efficient KV cache |
| **LLM inference (prefill)** | PA Prefill / MHA varlen | CK (C++) or Triton | High throughput for long contexts |
| **DeepSeek-style models** | MLA | ASM (decode) + CK (prefill) | Purpose-built for latent attention |
| **Mixed prefill+decode batches** | Unified Attention | Triton | Handles heterogeneous batches in one kernel |
| **Sparse/TopK attention** | Sparse MLA | Triton | Only backend with sparse support |
| **Prototyping / new GPUs** | Any Triton variant | Triton | Portable, easy to modify |

---

## 1. Multi-Head Attention (MHA / Flash Attention)

Standard scaled dot-product attention with multiple parallel heads. This is the foundational attention operation used in most transformer models.

### Backend Support

| Feature | CK (C++) | Triton | ASM |
|---------|:---:|:---:|:---:|
| **Forward pass** | Yes | Yes | Yes |
| **Backward pass** | Yes | Yes (varlen) | - |
| **BF16** | Yes | Yes | Yes |
| **FP16** | Yes | Yes | Yes |
| **FP8** | Yes | Yes | - |
| **Causal masking** | Yes | Yes | Yes |
| **ALiBi bias** | Yes | Yes | - |
| **Sliding window** | Yes | Yes | - |
| **Dropout** | Yes | Yes | - |
| **Variable-length (varlen)** | Yes | Yes | Yes |
| **GQA (any ratio)** | Yes | Yes | Yes |
| **GFX942 (MI300X)** | Yes | Yes | Yes |
| **GFX950 (MI350)** | Yes | Yes | Limited |

### Key API Functions

```python
from aiter.ops.triton.attention import (
    flash_attn_func,              # Standard forward
    flash_attn_varlen_func,       # Variable-length forward
    flash_attn_fp8_func,          # FP8 forward
    flash_attn_varlen_fp8_func,   # FP8 variable-length forward
)
```

### When to Use

- General-purpose attention for training and inference
- When you need backward pass support
- When you need ALiBi, sliding window, or dropout
- Models: GPT, LLaMA, Mistral, Qwen, and most standard transformers

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/mha.py` | Top-level MHA operations |
| `aiter/ops/triton/attention/mha.py` | Triton wrapper |
| `aiter/ops/triton/_triton_kernels/attention/mha.py` | Triton kernel implementation |
| `csrc/cpp_itfs/mha_fwd.cu` | CK forward pass |
| `csrc/cpp_itfs/mha_bwd.cu` | CK backward pass |

---

## 2. Paged Attention (PA)

Memory-efficient attention for LLM inference with block-wise KV cache storage. Eliminates memory fragmentation by storing KV cache in fixed-size pages.

### Two Execution Phases

| Phase | Description | Optimized For |
|-------|-------------|---------------|
| **PA Decode** | Single-token generation (autoregressive) | Latency (one token at a time) |
| **PA Prefill** | Multi-token context processing | Throughput (long prompt ingestion) |

### Backend Support

| Feature | CK (C++) | Triton | ASM (GFX942) |
|---------|:---:|:---:|:---:|
| **Decode V1** (short seq) | Yes | Yes | Yes |
| **Decode V2** (long seq, partitioned) | Yes | Yes | Yes |
| **Prefill** | Yes | Yes | - |
| **Extend (prefix caching)** | Yes | Yes | - |
| **Ragged layout** | Yes | - | - |
| **Persistent scheduling** | - | - | Yes |
| **Multi-Token Processing (MTP)** | - | - | Yes |
| **BF16 Q + BF16 KV** | Yes | Yes | Yes |
| **FP16 Q + FP16 KV** | Yes | Yes | - |
| **BF16 Q + FP8 KV** | Yes | Yes | Yes |
| **BF16 Q + INT8 KV** | Yes | - | Yes |
| **GQA ratios** | Any | Any | 8, 10, 16 |
| **Block sizes** | 8, 16, 128, 256 | Configurable | 16 |
| **GFX942 (MI300X)** | Yes | Yes | Yes |
| **GFX950 (MI350)** | Yes | Yes | Limited |

### KV Cache Quantization Methods

| Method | Description | Memory Savings | Accuracy Impact |
|--------|-------------|---------------|-----------------|
| `NO` | No quantization (BF16/FP16) | Baseline | None |
| `KV_8BIT_PER_TENSOR` | FP8 per-tensor scale | ~50% | Low |
| `KV_8BIT_PER_TOKEN` | FP8 per-token scale | ~50% | Very low |
| `KV_8BIT_PER_HEAD` | FP8/INT8 per-head scale | ~50% | Very low |
| `KV_4BIT_PER_TOKEN` | INT4 per-token scale | ~75% | Moderate |

For FP8 KV cache, three precision levels control accuracy vs speed:
- `high_precision=0` — Fastest, lowest accuracy
- `high_precision=1` — Balanced (default)
- `high_precision=2` — Slowest, highest accuracy

### Key API Functions

```python
from aiter.paged_attn import paged_attention_decode

# Decode phase (single-token generation)
paged_attention_decode(
    output, query, key_cache, value_cache,
    seq_lens, block_tables,
    attn_scale=1.0 / math.sqrt(head_dim),
    max_seq_len=max_context,
    k_scale=k_quant_scale,    # optional: for quantized KV cache
    v_scale=v_quant_scale,    # optional: for quantized KV cache
)

# Prefill phase (context processing)
from aiter.ops.triton.attention import context_attention_fwd
context_attention_fwd(query, key, value, output, ...)

# Extend with prefix caching
from aiter.ops.triton.attention import extend_attention_fwd
extend_attention_fwd(query, key, value, output, ...)
```

### Automatic Kernel Selection

AITER automatically selects the optimal backend based on workload:

```python
# Internal heuristic: ASM preferred for large batch * heads
total_heads = num_seqs * num_heads
use_asm = total_heads > 2 * num_compute_units
```

### When to Use

- LLM inference serving with dynamic batching (e.g., vLLM-style)
- When memory efficiency for KV cache is critical
- When serving many concurrent sequences
- Models: Any autoregressive LLM

### Source Files

| File | Purpose |
|------|---------|
| `aiter/paged_attn.py` | Public API |
| `aiter/ops/attention.py` | Wrapper and dispatch logic |
| `aiter/ops/triton/attention/pa_decode.py` | Triton decode wrapper |
| `aiter/ops/triton/attention/pa_prefill.py` | Triton prefill wrapper |
| `csrc/cpp_itfs/pa/pa.py` | C++ interface |
| `hsa/gfx942/pa/` | ASM kernel binaries |

---

## 3. Multi-head Latent Attention (MLA)

Latent projection-based attention from DeepSeek-V2/V3, which compresses KV representations into a low-rank latent space. Significantly reduces KV cache size and memory bandwidth requirements.

### Backend Support

| Feature | ASM | Triton | CK (C++) |
|---------|:---:|:---:|:---:|
| **Decode (seqlen_q=1)** | Yes (22+ kernels) | Yes | - |
| **Decode (seqlen_q=2,4,8)** | Yes | - | - |
| **Prefill** | Yes | - | - |
| **Persistent scheduling** | Yes | - | - |
| **Sparse (Top-K)** | - | Yes | - |
| **BF16 Q x BF16 KV** | Yes | Yes | - |
| **FP8 Q x FP8 KV** | Yes | - | - |
| **BF16 Q x FP8 KV** | Yes | - | - |
| **RoPE fusion** | Yes (native) | Yes (in-kernel) | - |
| **Arbitrary GQA ratio** | No (1,16,32,64,128) | Yes | - |
| **Page size > 1** | Yes | No (page_size=1 only) | - |
| **GFX942 (MI300X)** | Yes | Yes | - |
| **GFX950 (MI350)** | Yes | Yes | - |

### Key API Functions

```python
from aiter import mla_decode_fwd, mla_prefill_fwd

# Decode (single-token generation)
output, lse = mla_decode_fwd(
    q, kv_buffer, o,
    qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    max_seqlen_q=1, page_size=1,
    nhead_kv=num_heads,
    sm_scale=1.0 / math.sqrt(qk_head_dim),
)

# Prefill
output = mla_prefill_fwd(
    q, kv_buffer, o,
    qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    max_seqlen_q=seq_len, page_size=page_size,
    nhead_kv=num_heads,
    sm_scale=1.0 / math.sqrt(qk_head_dim),
)
```

### When to Use

- DeepSeek-V2, DeepSeek-V3, and other MLA-based models
- When KV cache memory is a bottleneck
- When latent compression is part of the model architecture

### Source Files

| File | Purpose |
|------|---------|
| `aiter/mla.py` | Main MLA interface (decode, prefill, persistent) |
| `aiter/ops/attention.py` | Metadata generation and reduce functions |
| `aiter/ops/triton/attention/mla_decode_rope.py` | Triton decode with RoPE |
| `aiter/ops/triton/attention/unified_attention_sparse_mla.py` | Triton sparse MLA |
| `csrc/kernels/mla/` | C++ metadata and reduce kernels |
| `hsa/gfx942/mla/` | ASM kernel binaries for MI300X |
| `hsa/gfx950/mla/` | ASM kernel binaries for MI350 |

---

## 4. Unified Attention (UA)

A flexible kernel that handles mixed prefill and decode batches in a single launch, avoiding the overhead of dispatching separate kernels.

### Backend Support

| Feature | Triton |
|---------|:---:|
| **Mixed prefill+decode** | Yes |
| **BF16, FP16** | Yes |
| **Causal masking** | Yes |
| **Sliding window** | Yes |
| **GQA (any ratio)** | Yes |
| **2D kernel** (shorter seqs) | Yes |
| **3D kernel** (longer seqs) | Yes |

### Key API Functions

```python
from aiter.ops.triton.attention import unified_attention

unified_attention(query, key, value, output, ...)
```

### When to Use

- Serving systems with continuous batching where prefill and decode requests coexist
- When you want to avoid separate kernel dispatches for prefill vs decode

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/triton/attention/unified_attention.py` | Wrapper |
| `aiter/ops/triton/_triton_kernels/attention/unified_attention.py` | Kernel |

---

## 5. Specialized Attention Variants

### Lean Attention

Persistent kernel variant for improved SM utilization with variable batch sizes.

- **Backend:** Triton
- **Features:** Paged KV cache, variable-length sequences
- **Source:** `aiter/ops/triton/attention/lean_atten.py`

### HSTU Attention (Hierarchical Softmax Truncated Unit)

Attention variant with SiLU activation on attention logits instead of softmax.

- **Backend:** Triton
- **Features:** SiLU activation, variable-length, forward and backward
- **Source:** `aiter/ops/triton/attention/hstu_attention.py`

### POD Attention

Specialized paged attention variant.

- **Backend:** Triton
- **Source:** `aiter/ops/triton/attention/pod_attention.py`

### Chunked Prefill with Paged Decode

Breaks long prefill sequences into chunks to interleave with decode operations.

- **Backend:** Triton
- **Source:** `aiter/ops/triton/attention/chunked_pa_prefill.py`

### FP8 MQA Logits (DeepGemm-style)

Advanced logits computation for FP8 multi-query attention.

- **Backend:** Triton
- **Source:** `aiter/ops/triton/attention/pa_mqa_logits.py`, `aiter/ops/triton/attention/fav3_sage.py`

---

## 6. Fused Operations with Attention

AITER provides several fused kernels that combine attention-adjacent operations to reduce memory round-trips.

### RoPE + Cache Fusion

| Operation | Backend | Description |
|-----------|---------|-------------|
| `fused_qk_rope_cat_and_cache_mla` | Triton | Fuse RoPE + Q/K concatenation + KV cache write |
| `fused_fp8_bmm_rope_cat_and_cache_mla` | Triton | + FP8 GEMM for latent projection |
| `fused_fp4_bmm_rope_cat_and_cache_mla` | Triton | + FP4 GEMM for latent projection |
| `fused_qk_rope_concat_and_cache_mla` | CK (C++) | CK version of RoPE + cache fusion |
| `fused_qk_norm_rope_cache_quant` | CK (C++) | + LayerNorm + quantization |

### Cache Management

```python
from aiter.ops.cache import (
    reshape_and_cache,                        # Standard cache update
    reshape_and_cache_flash,                  # Flash-optimized layout
    reshape_and_cache_with_pertoken_quant,    # Per-token FP8 cache
    reshape_and_cache_with_block_quant,       # Block-wise quantized cache
    concat_and_cache_mla,                     # MLA-specific cache
)
```

---

## 7. Backend Comparison: How to Choose

### Performance Characteristics

| Backend | Strengths | Weaknesses |
|---------|-----------|------------|
| **ASM** | Best raw performance, hand-tuned for MI300X/MI350 | Fixed configs, arch-specific, no backward pass |
| **CK (C++)** | Good performance, forward+backward, broad dtype | Requires ROCm, arch-specific compilation |
| **Triton** | Portable, easy to modify, flexible configs | Slightly lower peak perf than ASM |

### Decision Tree

```
Is this for training (need backward pass)?
├── Yes → Use MHA (CK C++)
└── No (inference only)
    ├── Standard transformer model?
    │   ├── Decode phase → Paged Attention (ASM if MI300X, else Triton)
    │   └── Prefill phase → PA Prefill (CK C++) or MHA varlen (Triton)
    ├── DeepSeek-style (MLA)?
    │   ├── Decode → MLA (ASM)
    │   ├── Prefill → MLA Prefill (ASM)
    │   └── Need sparse attention → Sparse MLA (Triton)
    ├── Mixed prefill+decode batches?
    │   └── Unified Attention (Triton)
    └── New/unsupported GPU?
        └── Any Triton variant
```

---

## 8. GPU Architecture Support Summary

| Attention Variant | MI300X (GFX942) | MI350 (GFX950) | Other GPUs |
|-------------------|:---:|:---:|:---:|
| MHA (Flash) | All backends | CK + Triton | Triton only |
| PA Decode | ASM + CK + Triton | CK + Triton | Triton only |
| PA Prefill | CK + Triton | CK + Triton | Triton only |
| MLA Decode | ASM + Triton | ASM + Triton | Triton only |
| MLA Prefill | ASM | ASM | - |
| Unified Attention | Triton | Triton | Triton |
| Sparse MLA | Triton | Triton | Triton |

---

## 9. Test Files Reference

| Test File | Covers |
|-----------|--------|
| `op_tests/test_mha.py` | MHA: BF16, FP16, FP8 |
| `op_tests/test_mha_fp8.py` | MHA FP8 precision |
| `op_tests/test_mha_varlen.py` | MHA variable-length sequences |
| `op_tests/test_mha_varlen_fp8.py` | MHA varlen + FP8 |
| `op_tests/test_pa.py` | PA decode + prefill, multiple dtypes |
| `op_tests/test_pa_v1.py` | PA V1 (short sequences) |
| `op_tests/test_pa_ps.py` | PA persistent scheduling |
| `op_tests/test_pa_mtp.py` | PA multi-token processing |
| `op_tests/test_pa_ragged.py` | PA ragged layout |
| `op_tests/test_mla.py` | MLA decode |
| `op_tests/test_mla_persistent.py` | MLA persistent mode |
| `op_tests/test_mla_prefill_ps.py` | MLA prefill |
| `op_tests/test_mla_sparse.py` | MLA sparse attention |
| `op_tests/test_batch_prefill.py` | Batch prefill |
| `op_tests/triton_tests/attention/` | All Triton attention tests |
