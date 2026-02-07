# AITER MLA (Multi-head Latent Attention) Variants & Backend Guide

This guide documents all MLA variants available in AITER, their backend support, and how to choose the right one for your use case.

---

## Quick Reference: Which MLA Configuration Should I Use?

| Use Case | Recommended Variant | Backend | Why |
|----------|-------------------|---------|-----|
| **DeepSeek-V2/V3 decode (production)** | Persistent Decode | ASM | Best latency, constant GPU occupancy |
| **DeepSeek-V2/V3 decode (single-token)** | Standard Decode | ASM | Mature, 22+ kernel variants |
| **DeepSeek-V2/V3 prefill** | Persistent Prefill | ASM | Causal mask, tile-based parallelism |
| **Multi-token prediction (MTP)** | Persistent Decode (seqlen_q=2,4) | ASM | Only backend with seqlen_q>1 |
| **FP8 inference (latency)** | Standard/Persistent Decode (FP8) | ASM | Native FP8xFP8 attention |
| **Sparse/TopK attention** | Sparse MLA | Triton | Only backend with sparse support |
| **FP4 latent projection** | Fused BMM+RoPE+Cache | Triton | Fused FP4 GEMM + cache write |
| **Custom GQA ratios** | Triton Decode RoPE | Triton | Arbitrary GQA via runtime param |
| **Prototyping / new GPUs** | Triton Decode RoPE | Triton | Portable, easy to modify |

---

## 1. MLA Architecture in AITER

MLA compresses Key-Value representations into a low-rank latent space, significantly reducing KV cache size and memory bandwidth. The KV cache stores a concatenated vector per token:

```
KV cache per token = [kv_latent (kv_lora_rank) | rope_embedding (qk_rope_head_dim)]
                      └─── typically 512 ───┘   └──── typically 64 ────┘

Standard MHA KV cache:  num_heads x head_dim  per token  (e.g., 128 x 128 = 16,384)
MLA KV cache:           kv_lora_rank + qk_rope_head_dim  (e.g., 512 + 64 = 576)
                        → ~28x compression
```

### Two-Stage Execution

All MLA backends use a split-K two-stage pipeline:

```
Stage 1: Parallel Attention                  Stage 2: Reduction
┌─────────────────────────┐                 ┌──────────────────┐
│ Split KV into K chunks  │                 │ Log-sum-exp      │
│ Each chunk computes:    │ ──────────────► │ reduction across  │
│   QK^T → softmax → V   │  partial        │ all K partials   │
│   + partial LSE         │  outputs        │ → final output   │
└─────────────────────────┘                 └──────────────────┘
     (ASM or Triton)                          (always Triton)
```

### Key Concepts

- **kv_lora_rank**: Dimension of the latent KV projection (typically 512)
- **qk_rope_head_dim**: Dimension of the RoPE component (typically 64)
- **Split-K**: KV sequence partitioned across CUs for parallelism (1-16 splits, auto-tuned)
- **Persistent mode**: Kernel stays resident on CUs with metadata-driven scheduling for better load balancing
- **Page size**: KV cache page granularity (1 = token-level, 64 = block-level)

---

## 2. Standard Decode (Non-Persistent)

Single-token autoregressive generation. The most common MLA operation for LLM inference serving.

### Backend Support

| Feature | ASM (GFX942/GFX950) | Triton |
|---------|:---:|:---:|
| **seqlen_q = 1** | Yes | Yes |
| **seqlen_q = 2, 4, 8** | Yes (GQA=16 only) | - |
| **BF16 Q x BF16 KV** | Yes | Yes |
| **FP8 Q x FP8 KV** | Yes (GQA=16, 128) | - |
| **BF16 Q x FP8 KV** | Yes (GQA=16, persistent) | - |
| **RoPE fusion** | Yes (native in-kernel) | Yes (in-kernel) |
| **Logit capping** | - | Yes |
| **GQA ratios** | 16, 32, 64, 128 | Any (runtime param) |
| **Page sizes** | 1, 64 | 1 only |
| **Split-K (auto-tuned)** | 1-16 | 1-16 |
| **GFX942 (MI300X)** | Yes | Yes |
| **GFX950 (MI350)** | Yes | Yes |
| **Other GPUs** | - | Yes (portable) |

### Key API Functions

```python
from aiter import mla_decode_fwd

# ASM decode (production path)
output, lse = mla_decode_fwd(
    q,                          # [total_s, num_heads, qk_head_dim]
    kv_buffer,                  # [num_pages, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    o,                          # [total_s, num_heads, v_head_dim]
    qo_indptr,                  # [batch_size + 1]
    kv_indptr,                  # [batch_size + 1]
    kv_indices,                 # page indices
    kv_last_page_lens,          # [batch_size]
    max_seqlen_q=1,
    page_size=1,
    nhead_kv=1,                 # number of KV heads
    sm_scale=1.0 / math.sqrt(qk_head_dim),
)
```

```python
from aiter.ops.triton.attention.mla_decode_rope import decode_attention_fwd_grouped_rope

# Triton decode with RoPE (flexible path)
output = decode_attention_fwd_grouped_rope(
    q, k_buffer, v_buffer, o,
    kv_indptr, kv_indices,
    k_pe_tokens,                # keys with RoPE applied
    kv_lora_rank=512,
    rotary_dim=64,
    cos_sin_cache=cos_sin_cache,
    positions=positions,
    attn_logits=attn_logits,    # intermediate buffer
    num_kv_splits=8,
    sm_scale=sm_scale,
    use_rope=True,
    is_neox_style=True,
)
```

### When to Use

- LLM inference serving with DeepSeek-V2/V3 or similar MLA models
- Standard autoregressive token generation (seqlen_q=1)
- Multi-token prediction with seqlen_q=2,4 (ASM only)
- Models: DeepSeek-V2, DeepSeek-V3, DeepSeek-R1

### Source Files

| File | Purpose |
|------|---------|
| `aiter/mla.py` | Main MLA interface (decode, prefill, persistent) |
| `aiter/ops/attention.py` | ASM kernel dispatch and metadata |
| `aiter/ops/triton/attention/mla_decode_rope.py` | Triton decode wrapper |
| `aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py` | Triton kernel implementation |

---

## 3. Persistent Decode

Persistent-mode decode keeps kernels resident on CUs with metadata-driven scheduling. Provides better load balancing across variable-length sequences.

### Backend Support

| Feature | ASM (GFX942/GFX950) |
|---------|:---:|
| **seqlen_q = 1** | Yes |
| **seqlen_q = 2, 4** | Yes |
| **BF16 Q x BF16 KV** | Yes |
| **FP8 Q x FP8 KV** | Yes (GQA=16, 128) |
| **BF16 Q x FP8 KV** | Yes (GQA=16) |
| **GQA = 16** | Yes (native) |
| **GQA = 32-112 (step 16)** | Yes (simulated via metadata reshaping) |
| **GQA = 128** | Yes (FP8 only, native kernel) |
| **Page sizes** | 1, 64 |
| **Intra-batch mode** | Yes |

### Key API Functions

```python
from aiter import mla_decode_fwd
from aiter.ops.attention import get_mla_metadata_v1

# Step 1: Generate persistent-mode metadata
metadata = get_mla_metadata_v1(
    seqlens_qo_indptr, seqlens_kv_indptr,
    kv_last_page_lens,
    page_size=1,
    max_seqlen_qo=1,
    dtype_q=torch.bfloat16,
    dtype_kv=torch.bfloat16,
)
work_meta_data, work_indptr, work_info_set, \
    reduce_indptr, reduce_final_map, reduce_partial_map = metadata

# Step 2: Run persistent decode
output, lse = mla_decode_fwd(
    q, kv_buffer, o,
    qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    max_seqlen_q=1,
    nhead_kv=1,
    sm_scale=sm_scale,
    work_meta_data=work_meta_data,
    work_indptr=work_indptr,
    work_info_set=work_info_set,
    reduce_indptr=reduce_indptr,
    reduce_final_map=reduce_final_map,
    reduce_partial_map=reduce_partial_map,
)
```

### When to Use

- Production inference with highly variable sequence lengths (better load balancing)
- When GPU occupancy matters (persistent kernels avoid launch overhead)
- Multi-token prediction (MTP) with seqlen_q > 1

### Source Files

| File | Purpose |
|------|---------|
| `aiter/mla.py` | Persistent decode logic (lines 283-350) |
| `aiter/ops/attention.py` | Metadata generation (`get_mla_metadata_v1`) |
| `csrc/kernels/mla/` | C++ metadata and reduce kernels |

---

## 4. Prefill

Context processing for the prompt phase. Two sub-variants: standard prefill and persistent prefill.

### Backend Support

| Feature | ASM Standard Prefill | ASM Persistent Prefill |
|---------|:---:|:---:|
| **BF16 Q x BF16 KV** | Yes | Yes |
| **FP8 Q x FP8 KV** | Yes (GQA=1, 128) | Yes |
| **Causal masking** | Yes | Yes |
| **Non-causal** | Yes (FP8, GQA=1) | - |
| **GQA = 1** | Yes | Yes (192/128 variant) |
| **GQA = 16** | Yes | - |
| **GQA = 128** | Yes | - |
| **3-buffer KV cache** | - | Yes (nope + scale + rope) |
| **Tile-based parallelism** | - | Yes (256-token tiles) |

### Key API Functions

```python
from aiter import mla_prefill_fwd

# Standard prefill
output, lse = mla_prefill_fwd(
    q, kv_buffer, o,
    qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    max_seqlen_q=seq_len,
    sm_scale=sm_scale,
)
```

```python
from aiter import mla_prefill_ps_fwd

# Persistent prefill (with causal masking)
output, lse = mla_prefill_ps_fwd(
    Q, K, V, output,
    qo_indptr, kv_indptr, kv_page_indices,
    work_indptr, work_info_set,
    max_seqlen_q=seq_len,
    is_causal=True,
    reduce_indptr=reduce_indptr,
    reduce_final_map=reduce_final_map,
    reduce_partial_map=reduce_partial_map,
    softmax_scale=sm_scale,
)
```

### 3-Buffer KV Cache Layout (Persistent Prefill with FP8)

For FP8 persistent prefill, the KV cache splits into three separate buffers:

```
kv_nope_buffer_fp8:         [num_page, page_size, 1, kv_lora_rank]       # FP8 latent
kv_nope_scale_factors_fp32: [num_page, page_size, 1, scale_dim]          # FP32 scales
kv_rope_buffer_bf16:        [num_page, page_size, 1, qk_rope_head_dim]   # BF16 rope
```

### When to Use

- Prompt ingestion for DeepSeek-style models
- Persistent prefill preferred for long contexts (better parallelism)
- 3-buffer layout required for FP8 persistent prefill

### Source Files

| File | Purpose |
|------|---------|
| `aiter/mla.py` | Prefill functions (standard + persistent) |
| `aiter/ops/attention.py` | `mla_prefill_asm_fwd`, `mla_prefill_ps_asm_fwd` |

---

## 5. Sparse MLA (Top-K Attention)

Sparse attention using top-K token selection. Reduces computation by only attending to the K most relevant tokens per query.

### Backend Support

| Feature | Triton |
|---------|:---:|
| **BF16** | Yes |
| **Top-K selection** | Yes (configurable K) |
| **Block tables** | Yes |
| **Variable-length** | Yes |
| **GQA** | Yes (single KV head) |

### Key API Functions

```python
from aiter.ops.triton.attention import unified_attention_sparse_mla

unified_attention_sparse_mla(
    q,                  # [seq_len, NUM_HEADS, kv_lora_rank + rope_rank]
    kv,                 # [seq_len_kv, 1, kv_lora_rank + rope_rank]
    out,                # [seq_len, NUM_HEADS, kv_lora_rank]
    cu_seqlens_q,       # [BATCH + 1]
    max_seqlen_q,
    seqused_k,          # [BATCH]
    max_seqlen_k,
    softmax_scale,
    topk_indices,       # [seq_len, TOP_K]
    block_table,        # [BATCH, MAX_NUM_BLOCKS_PER_BATCH]
    kv_lora_rank=512,
)
```

### When to Use

- When full attention is too expensive and approximate attention is acceptable
- Speculative decoding with sparse token selection
- Research on efficient MLA attention patterns

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/triton/attention/unified_attention_sparse_mla.py` | Sparse MLA wrapper |
| `aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py` | Triton kernel |

---

## 6. Fused Operations with MLA

AITER provides fused kernels that combine MLA-adjacent operations to reduce memory round-trips and kernel launch overhead.

### RoPE + Cache Fusion

| Operation | Backend | Description |
|-----------|---------|-------------|
| `concat_and_cache_mla` | ASM | Concatenate kv_latent + k_rope into paged cache |
| `fused_qk_rope_concat_and_cache_mla` | ASM | Fuse RoPE + Q/K concatenation + cache write |
| `fused_qk_rope_cat_and_cache_mla` | Triton | Triton version of RoPE + cache fusion |
| `fused_fp8_bmm_rope_cat_and_cache_mla` | Triton | + FP8 GEMM for latent projection |
| `fused_fp4_bmm_rope_cat_and_cache_mla` | Triton | + FP4 (MXFP4) GEMM for latent projection |

### Fused BMM + RoPE + Cache (Triton)

Combines the latent projection GEMM (FP4/FP8), RoPE application, and KV cache write into a single kernel launch:

```python
from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
    fused_fp4_bmm_rope_cat_and_cache_mla,
)

# FP4 latent projection + RoPE + cache write in one kernel
q_out, decode_q_pe_out, k_pe_out, _ = fused_fp4_bmm_rope_cat_and_cache_mla(
    q_nope,             # (QH, B, P)
    w_k,                # FP4 weights: (QH, kv_lora_rank, P//2)
    w_k_scale,          # E8M0 scales: (QH, kv_lora_rank, P//32)
    q_pe, k_nope, k_rope,
    kv_cache, slot_mapping,
    positions, cos, sin,
)
```

### ASM Cache Operations

```python
from aiter.ops.cache import (
    concat_and_cache_mla,                       # Basic MLA cache write
    fused_qk_rope_concat_and_cache_mla,        # + RoPE fusion
)

# Basic: concatenate latent + rope and write to paged cache
concat_and_cache_mla(
    kv_c,               # [num_tokens, kv_heads, kv_lora_rank]
    k_pe,               # [num_tokens, kv_heads, qk_rope_head_dim]
    kv_cache,            # [num_pages, page_size, kv_heads, kv_lora_rank + qk_rope_head_dim]
    slot_mapping,
    kv_cache_dtype="auto",
    scale=1.0,
)

# Fused: RoPE + concat + cache in one kernel
fused_qk_rope_concat_and_cache_mla(
    q_nope, q_pe, kv_c, k_pe,
    kv_cache, q_out,
    slot_mapping, k_scale, q_scale,
    positions, cos_cache, sin_cache,
    is_neox=True, is_nope_first=True,
)
```

### When to Use

- Always prefer fused operations over separate RoPE + cache write
- FP4/FP8 fused BMM is unique to Triton and not available in ASM
- ASM fused cache is preferred for production BF16/FP8 workloads

### Source Files

| File | Purpose |
|------|---------|
| `aiter/ops/cache.py` | ASM cache operations |
| `aiter/ops/triton/fusions/fused_kv_cache.py` | Triton RoPE + cache |
| `aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py` | Triton BMM + RoPE + cache |

---

## 7. Data Type & Quantization Support Matrix

### By Backend and Variant

| Configuration | ASM Decode | ASM Prefill | Triton Decode | Triton Fused BMM |
|--------------|:---:|:---:|:---:|:---:|
| **BF16 Q x BF16 KV** | Yes | Yes | Yes | Yes (activation) |
| **FP8 Q x FP8 KV** | Yes (GQA=16,128) | Yes (GQA=1,128) | - | Yes (GEMM weights) |
| **BF16 Q x FP8 KV** | Yes (GQA=16, persistent) | - | - | - |
| **FP4 (MXFP4) GEMM** | - | - | - | Yes |
| **FP16** | - | - | Yes (implicit) | - |
| **FP32 accumulator** | - | - | Yes | - |

### Choosing the Right Data Type

```
Need maximum accuracy?
├── Yes → BF16 Q x BF16 KV (any backend)
└── No
    ├── Need reduced KV cache memory?
    │   ├── FP8 Q x FP8 KV → ASM (native attention)
    │   └── BF16 Q x FP8 KV → ASM persistent (mixed precision)
    ├── Need compressed latent projection?
    │   ├── FP8 GEMM + BF16 attention → Triton fused BMM
    │   └── FP4 GEMM + BF16 attention → Triton fused BMM (max compression)
    └── Need portability?
        └── BF16 Triton decode (works on any GPU)
```

---

## 8. GQA (Grouped Query Attention) Ratio Support

| GQA Ratio | ASM Decode | ASM Persistent | ASM Prefill | Triton |
|-----------|:---:|:---:|:---:|:---:|
| **1 (full MHA)** | - | - | Yes (FP8 prefill) | Yes |
| **16** | Yes (all dtypes) | Yes (native) | Yes | Yes |
| **32** | Yes (BF16 only) | Yes (simulated) | - | Yes |
| **48, 64, 80, 96, 112** | Yes (BF16, select) | Yes (simulated, step 16) | - | Yes |
| **128** | Yes (FP8 only) | Yes (FP8 native) | Yes | Yes |
| **Arbitrary** | - | - | - | Yes (runtime param) |

**Note:** ASM persistent mode simulates GQA ratios 32-112 by viewing `nhead_kv` heads as `nhead_kv * (gqa/16)` heads with adjusted dimensions, then dispatching to the GQA=16 kernel.

---

## 9. KV Cache Layouts

### Standard Layout (All Variants)

```
kv_cache: [num_pages, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
```

### 3-Buffer Layout (FP8 Persistent Prefill)

```
kv_nope_buffer_fp8:         [num_pages, page_size, 1, kv_lora_rank]        # FP8
kv_nope_scale_factors_fp32: [num_pages, page_size, 1, scale_dim]           # FP32
kv_rope_buffer_bf16:        [num_pages, page_size, 1, qk_rope_head_dim]    # BF16
```

### Page Size Support

| Page Size | ASM | Triton Decode | Triton Fused Cache |
|-----------|:---:|:---:|:---:|
| **1** (token-level) | Yes | Yes | Yes |
| **64** (block-level) | Yes (select kernels) | - | Yes |

---

## 10. RoPE (Rotary Position Encoding) Support

### Styles

| Style | Description | Backend Support |
|-------|-------------|----------------|
| **GPT-J** (block) | Pairs consecutive dimensions: (0,1), (2,3), ... | ASM + Triton |
| **NeoX** (interleaved) | Pairs split dimensions: (0, N/2), (1, N/2+1), ... | ASM + Triton |

### RoPE Application Points

1. **In-kernel** (during attention): Both ASM and Triton fuse RoPE into the attention kernel, applying it to q_pe and k_pe before QK^T
2. **During caching** (fused cache write): `fused_qk_rope_concat_and_cache_mla` (ASM) and `fused_qk_rope_cat_and_cache_mla` (Triton) apply RoPE while writing to the KV cache

### Configuration

```python
# RoPE parameters
qk_rope_head_dim = 64           # rotary dimension
cos_sin_cache.shape              # [max_seq_len, rotary_dim] or [max_seq_len, rotary_dim//2]
is_neox_style = True             # NeoX vs GPT-J
```

---

## 11. Backend Comparison: How to Choose

### Performance Characteristics

| Backend | Strengths | Weaknesses |
|---------|-----------|------------|
| **ASM** | Best raw performance, decode+prefill+persistent, FP8 native | Fixed configs, arch-specific, no portability |
| **Triton** | Portable, arbitrary GQA, sparse support, FP4 GEMM fusion | No prefill, seqlen_q=1 only, page_size=1 only |

### Decision Tree

```
Is this for prefill (context processing)?
├── Yes → ASM Prefill (standard or persistent)
└── No (decode / token generation)
    ├── Need persistent scheduling (production)?
    │   └── ASM Persistent Decode
    ├── Need multi-token prediction (seqlen_q > 1)?
    │   └── ASM Decode (seqlen_q=2,4,8)
    ├── Need sparse / Top-K attention?
    │   └── Triton Sparse MLA
    ├── Need FP4 latent projection?
    │   └── Triton Fused BMM+RoPE+Cache
    ├── Need arbitrary GQA ratio?
    │   └── Triton Decode RoPE
    ├── Need portability (non-MI300X/MI350)?
    │   └── Triton Decode RoPE
    └── Standard decode (production)?
        └── ASM Standard Decode
```

---

## 12. GPU Architecture Support Summary

| MLA Variant | MI300X (GFX942) | MI350 (GFX950) | Other GPUs |
|-------------|:---:|:---:|:---:|
| Standard Decode (ASM) | Yes (20+ kernels) | Yes (24+ kernels) | - |
| Persistent Decode (ASM) | Yes | Yes | - |
| Standard Prefill (ASM) | Yes | Yes | - |
| Persistent Prefill (ASM) | Yes | Yes | - |
| Triton Decode RoPE | Yes (tuned) | Yes (tuned) | Yes (portable) |
| Sparse MLA (Triton) | Yes | Yes | Yes (portable) |
| Fused BMM+Cache (Triton) | Yes | Yes | Yes (portable) |

### ASM Kernel Inventory

| Architecture | Decode Kernels | Persistent Kernels | Prefill Kernels | Total |
|-------------|:-:|:-:|:-:|:-:|
| **GFX942 (MI300X)** | ~8 | ~8 | ~4 | ~20 |
| **GFX950 (MI350)** | ~10 | ~10 | ~4 | ~24 |

Kernel configurations are cataloged in `hsa/gfx942/mla/mla_asm.csv` and `hsa/gfx950/mla/mla_asm.csv` with columns: `qType, kvType, Gqa, ps, qSeqLen, prefill, causal, knl_name, co_name`.

---

## 13. Performance Tuning

### Split-K Auto-Tuning

MLA automatically selects the optimal split-K count based on:

```python
# Factors considered:
# - Batch size (bs)
# - Total KV sequence length (total_kv)
# - Number of CUs
# - Data type
# - Overhead constant (empirically 84.1)

# Tests splits 1-16 and picks best using:
# throughput = (bs * i) / ceil(bs * i / cu_num) * avg_kv / (avg_kv + overhead * i)
```

Override by passing `num_kv_splits` explicitly to `mla_decode_fwd()`.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AITER_LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | INFO |
| `AITER_LOG_MORE` | Enable detailed logging with line numbers | 0 |
| `AITER_TRITON_CONFIGS_PATH` | Path to Triton kernel config JSONs | Built-in |

### Key Parameters

- **num_kv_splits**: Split-K count (1-16). Higher values improve parallelism for long sequences but add reduction overhead.
- **page_size**: KV cache page granularity. page_size=1 (default) for token-level, page_size=64 for block-level (fewer metadata, better for long contexts).
- **intra_batch_mode**: When True, enables intra-batch parallelism for persistent mode (useful when batch size < num_CUs).
- **fast_mode**: Metadata generation mode. True (default) optimizes for speed.

---

## 14. Test Files Reference

| Test File | Covers |
|-----------|--------|
| `op_tests/test_mla.py` | Standard decode: BF16/FP8, GQA=16/128, variable contexts |
| `op_tests/test_mla_persistent.py` | Persistent decode: BF16/FP8, GQA=16/128, page_size=1/64 |
| `op_tests/test_mla_sparse.py` | Sparse attention: Top-K, block tables, FP8 |
| `op_tests/test_mla_prefill_ps.py` | Persistent prefill: causal, 3-buffer, FP8 (GFX950) |
| `op_tests/test_concat_cache_mla.py` | Cache concatenation operations |
| `op_tests/triton_tests/attention/test_mla_decode_rope.py` | Triton RoPE decode, split-K, paged cache |
| `op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py` | Triton sparse: Top-K, block tables |
| `op_tests/triton_tests/utils/mla_decode_ref.py` | Reference decode implementation |
| `op_tests/triton_tests/utils/mla_extend_ref.py` | Reference extend/prefill implementation |

### Benchmarks

| Benchmark File | Coverage |
|----------------|----------|
| `op_tests/op_benchmarks/triton/bench_mla_decode.py` | Model-level decode performance |
| `op_tests/op_benchmarks/triton/bench_mla_decode_rope.py` | RoPE decode performance |

---

## 15. Source Files Reference

### Main API

| File | Purpose |
|------|---------|
| `aiter/mla.py` | Primary MLA API: decode, prefill, persistent, reduction |
| `aiter/ops/attention.py` | ASM kernel dispatch, metadata generation, reduce functions |
| `aiter/ops/cache.py` | KV cache operations (concat, fused RoPE + cache) |

### Triton Kernels

| File | Purpose |
|------|---------|
| `aiter/ops/triton/attention/mla_decode_rope.py` | Triton decode with RoPE wrapper |
| `aiter/ops/triton/attention/unified_attention_sparse_mla.py` | Sparse MLA wrapper |
| `aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py` | Triton decode kernel implementation |
| `aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py` | Sparse MLA kernel implementation |

### Fused Operations (Triton)

| File | Purpose |
|------|---------|
| `aiter/ops/triton/fusions/fused_kv_cache.py` | RoPE + KV cache fusion |
| `aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py` | FP4/FP8 BMM + RoPE + cache fusion |
| `aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py` | Fused cache kernel |
| `aiter/ops/triton/_triton_kernels/fusions/fused_bmm_rope_kv_cache.py` | Fused BMM kernel |

### ASM Kernels

| File | Purpose |
|------|---------|
| `hsa/gfx942/mla/mla_asm.csv` | MI300X kernel dispatch catalog |
| `hsa/gfx942/mla/*.co` | MI300X compiled kernel binaries |
| `hsa/gfx950/mla/mla_asm.csv` | MI350 kernel dispatch catalog |
| `hsa/gfx950/mla/*.co` | MI350 compiled kernel binaries |
| `csrc/kernels/mla/` | C++ metadata and reduce kernels |

### AOT Compilation

| File | Purpose |
|------|---------|
| `aiter/aot/triton/decode_mla.py` | Ahead-of-time compilation for Triton MLA |
