# AITER KV-Cache Management Guide

This guide documents all KV-cache operators available in AITER, including cache layouts, quantized cache support, fused operations, and integration with attention kernels.

---

## Quick Reference: Which Cache Operation Should I Use?

| Use Case | Recommended Operation | Backend | Why |
|----------|---------------------|---------|-----|
| **Standard paged attention** | `reshape_and_cache` | CUDA/HIP | Default write to paged KV cache |
| **Flash attention layout** | `reshape_and_cache_flash` | CUDA/HIP | Contiguous [block, page, head, dim] layout |
| **FP8 quantized cache** | `reshape_and_cache_with_pertoken_quant` | CUDA/HIP | Per-token quantize during write |
| **Block-quantized cache** | `reshape_and_cache_with_block_quant` | CUDA/HIP | Block-level quantize for ASM PA |
| **MLA models (DeepSeek)** | `concat_and_cache_mla` | CUDA/HIP | Concatenates kv_c + k_pe |
| **MLA + RoPE fused** | `fused_qk_rope_cat_and_cache_mla` | Triton | RoPE + concat + cache in one kernel |
| **MLA + quantized BMM** | `fused_fp8_bmm_rope_cat_and_cache_mla` | Triton | BMM + RoPE + cache fully fused |
| **Cache migration** | `swap_blocks` | CUDA/HIP | Block-level swap (H2D, D2H, D2D) |
| **Multi-layer copy** | `copy_blocks` | CUDA/HIP | Copy blocks across cache layers |

---

## 1. Cache Layouts

### Paged Cache Layout (Default)

The standard layout for paged attention. Key and value caches use different memory layouts optimized for their access patterns.

```
Key cache:   [num_blocks, num_heads, head_size/x, block_size, x]
Value cache: [num_blocks, num_heads, head_size, block_size]

where x = 16 / sizeof(dtype)
  - x = 8  for BF16/FP16
  - x = 16 for INT8/FP8
```

The key cache uses a vectorized layout (the innermost `x` dimension) for cache-line-friendly memory access during attention dot products.

### ASM Layout

An alternative value cache layout optimized for ASM (hand-tuned assembly) attention kernels:

```
Key cache:   [num_blocks, num_heads, head_size/x, block_size, x]  (same)
Value cache: [num_blocks, num_heads, block_size/x, head_size, x]  (transposed)
```

Enable with `asm_layout=True` in cache write operations.

### Flash Attention Layout

A contiguous layout compatible with flash attention implementations:

```
Key cache:   [num_blocks, block_size, num_heads, head_size]
Value cache: [num_blocks, block_size, num_heads, head_size]
```

### MLA Cache Layout

Multi-head Latent Attention uses a unified cache storing concatenated KV projections:

```
KV cache: [num_blocks, block_size, kv_lora_rank + pe_dim]
# or multi-head variant:
KV cache: [num_blocks, block_size, num_kv_heads, kv_lora_rank + pe_dim]
```

---

## 2. Core Cache Write Operations

### Standard Write

```python
import aiter

# Write new KV pairs into paged cache
aiter.reshape_and_cache(
    key,          # [num_tokens, num_heads, head_size]
    value,        # [num_tokens, num_heads, head_size]
    key_cache,    # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache,  # [num_blocks, num_heads, head_size, block_size]
    slot_mapping, # [num_tokens] -> block slot index
    kv_cache_dtype="auto",  # "auto", "fp8", "fp8_e4m3"
    k_scale=None,           # Optional FP8 scale
    v_scale=None,           # Optional FP8 scale
    asm_layout=False,       # Use ASM value cache layout
)
```

### Flash Layout Write

```python
aiter.reshape_and_cache_flash(
    key, value,
    key_cache,    # [num_blocks, block_size, num_heads, head_size]
    value_cache,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping,
    kv_cache_dtype="auto",
    k_scale=None, v_scale=None,
)
```

### Per-Token Quantized Write

Quantizes KV pairs per-token during cache write, computing dequantization scales on the fly:

```python
aiter.reshape_and_cache_with_pertoken_quant(
    key, value,
    key_cache,          # Quantized cache (FP8/INT8)
    value_cache,        # Quantized cache (FP8/INT8)
    k_dequant_scales,   # [num_heads, max_tokens] per-token scales
    v_dequant_scales,   # [num_heads, max_tokens] per-token scales
    slot_mapping,
    asm_layout=False,
)
```

### Block-Level Quantized Write

Quantizes at block granularity for better scale sharing:

```python
aiter.reshape_and_cache_with_block_quant(
    key, value,
    key_cache, value_cache,
    k_dequant_scales,   # [num_heads, num_blocks]
    v_dequant_scales,   # [num_heads, num_blocks]
    slot_mapping,
    asm_layout=False,
)

# ASM PA variant with configurable block size
aiter.reshape_and_cache_with_block_quant_for_asm_pa(
    key, value,
    key_cache, value_cache,
    k_dequant_scales, v_dequant_scales,
    slot_mapping,
    asm_layout=True,
    ori_block_size=128,  # 128 or 256
)
```

---

## 3. MLA Cache Operations

### Basic MLA Concat + Cache

```python
# Concatenate kv_c (latent) and k_pe (positional encoding), write to cache
aiter.concat_and_cache_mla(
    kv_c,          # [num_tokens, kv_lora_rank]
    k_pe,          # [num_tokens, pe_dim]
    kv_cache,      # [num_blocks, block_size, kv_lora_rank + pe_dim]
    slot_mapping,
    kv_cache_dtype="auto",
    scale=None,     # Optional FP8 scale
)
```

### Fused RoPE + Concat + MLA Cache

Applies RoPE to Q and K, concatenates, and writes to cache in a single kernel:

```python
aiter.fused_qk_rope_concat_and_cache_mla(
    q_nope,       # [num_tokens, num_heads, nope_dim]
    q_pe,         # [num_tokens, num_heads, pe_dim]
    kv_c,         # [num_tokens, kv_lora_rank]
    k_pe,         # [num_tokens, pe_dim]
    kv_cache,     # [num_blocks, block_size, kv_lora_rank + pe_dim]
    q_out,        # [num_tokens, num_heads, qk_lora_rank + pe_dim]
    slot_mapping,
    k_scale, q_scale,
    positions,     # [num_tokens] position IDs
    cos_cache, sin_cache,  # RoPE cos/sin tables
    is_neox=True,
    is_nope_first=False,
)
```

### Indexed K Quantization + Cache

SGLang-style indexed cache write with per-block quantization:

```python
# Write K with quantization at arbitrary slots
aiter.indexer_k_quant_and_cache(
    k,             # [num_tokens, head_dim]
    kv_cache,      # Cache tensor
    slot_mapping,
    quant_block_size=64,
    scale_fmt="ue8m0",    # or "" for direct float
)

# Gather K from cache with dequantization
aiter.cp_gather_indexer_k_quant_cache(
    kv_cache,
    dst_k,         # Output tensor
    dst_scale,     # Output scales
    block_table,   # [batch, max_blocks_per_seq]
    cu_seq_lens,   # Cumulative sequence lengths
)
```

---

## 4. Cache Management Operations

### Block Swap

Swaps cache blocks between devices (host-to-device, device-to-host, device-to-device):

```python
aiter.swap_blocks(
    src,            # Source cache tensor
    dst,            # Destination cache tensor
    block_mapping,  # [N, 2] tensor of (src_block, dst_block) pairs
)
```

### Block Copy

Copies blocks across multiple cache layers (used for beam search, speculative decoding):

```python
aiter.copy_blocks(
    key_caches,     # List of key cache tensors (all layers)
    value_caches,   # List of value cache tensors (all layers)
    block_mapping,  # [N, 2] tensor of (src_block, dst_block) pairs
)
```

---

## 5. Triton Fused Cache Operations

### Fused QK RoPE + Reshape + Cache (Standard Attention)

```python
from aiter.ops.triton.fusions.fused_kv_cache import (
    fused_qk_rope_reshape_and_cache,
    fused_qk_rope_cosine_cache_llama,
)

# RoPE + reshape + cache write in one kernel
fused_qk_rope_reshape_and_cache(...)
```

### Fused QK RoPE + Concat + MLA Cache

```python
from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_cat_and_cache_mla

# Triton implementation of RoPE + MLA concat + cache
fused_qk_rope_cat_and_cache_mla(...)
```

### Fused BMM + RoPE + MLA Cache (Quantized)

Maximum fusion for quantized MLA inference - combines matrix multiply with RoPE application and cache write:

```python
from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
    fused_fp8_bmm_rope_cat_and_cache_mla,  # FP8 quantized BMM
    fused_fp4_bmm_rope_cat_and_cache_mla,  # FP4/MXFP4 quantized BMM
)
```

---

## 6. Quantized Cache Support

### Supported Quantization Types

| Cache Dtype | Key Cache | Value Cache | Scale Shape | Notes |
|------------|-----------|-------------|-------------|-------|
| **auto** (BF16/FP16) | Same as input | Same as input | None | No quantization |
| **FP8 (e4m3fn)** | FP8 | FP8 | Per-token or per-block | GFX950 (MI350) |
| **FP8 (e4m3fnuz)** | FP8 | FP8 | Per-token or per-block | GFX942 (MI300X) |
| **INT8** | INT8 | INT8 | Per-token or per-block | Universal support |
| **FP4 (MXFP4)** | FP4 | FP4 | E8M0 block scales | GFX950 only |

### Scale Tensor Shapes

| Quantization Mode | Scale Shape | Description |
|------------------|-------------|-------------|
| Per-token (non-ASM) | `[num_heads, max_tokens]` | One scale per head per token |
| Per-block | `[num_heads, num_blocks]` | One scale per head per block |
| Per-block (ASM) | `[num_blocks, num_heads, block_size]` | Transposed for ASM access |
| UE8M0 (MXFP4) | Block-based | Saturating FP32 format |

---

## 7. Integration with Attention Operators

### Paged Attention

Cache is read during attention computation via block table indirection:

```python
# PA reads from paged KV cache
attention_output = aiter.paged_attention_v1(
    query,
    key_cache,           # Paged key cache
    value_cache,         # Paged value cache
    block_tables,        # [batch, max_blocks_per_seq]
    seq_lens,
    k_dequant_scales,    # For quantized cache
    v_dequant_scales,
)
```

### MLA Decode

MLA reads from unified KV cache with separate latent and positional components:

```python
output = aiter.mla_decode_fwd(
    query, kv_buffer, kv_indptr, kv_indices,
    ...
)
```

### Page Table Formats

| Format | Shape | Used By |
|--------|-------|---------|
| VLLM Block Table | `[batch_size, max_blocks_per_seq]` | vLLM, most frameworks |
| SGLang Page Table | 1D with `kv_indptr` indexing | SGLang |

---

## 8. Decision Tree

```
Need to write KV to cache?
├── Standard attention model?
│   ├── No quantization → reshape_and_cache()
│   ├── FP8 per-token → reshape_and_cache_with_pertoken_quant()
│   ├── FP8 per-block → reshape_and_cache_with_block_quant()
│   └── Flash layout → reshape_and_cache_flash()
├── MLA model (DeepSeek)?
│   ├── Basic → concat_and_cache_mla()
│   ├── With RoPE → fused_qk_rope_concat_and_cache_mla()
│   └── With BMM + RoPE → fused_fp8_bmm_rope_cat_and_cache_mla()
├── Need block management?
│   ├── Swap blocks (beam search) → swap_blocks()
│   └── Copy blocks (speculative) → copy_blocks()
└── SGLang indexed access?
    ├── Write → indexer_k_quant_and_cache()
    └── Read → cp_gather_indexer_k_quant_cache()
```

---

## 9. Source Files

| Component | Path |
|-----------|------|
| Cache Python API | `aiter/ops/cache.py` |
| CUDA/HIP cache kernels | `csrc/kernels/cache_kernels.cu` |
| Cache header | `csrc/include/cache.h` |
| Triton fused KV cache | `aiter/ops/triton/fusions/fused_kv_cache.py` |
| Triton KV cache kernels | `aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py` |
| Triton fused BMM + cache | `aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py` |
| Fused QK Norm + RoPE + Cache | `aiter/ops/fused_qk_norm_rope_cache_quant.py` |
| MLA decode | `aiter/mla.py` |
| MLA Triton decode | `aiter/ops/triton/attention/mla_decode_rope.py` |

---

## 10. Test Files

| Test | Path |
|------|------|
| Basic cache ops | `op_tests/test_kvcache.py` |
| Block-scale cache | `op_tests/test_kvcache_blockscale.py` |
| MLA concat cache | `op_tests/test_concat_cache_mla.py` |
| Indexed K quant cache | `op_tests/test_indexer_k_quant_and_cache.py` |
| Triton fused KV cache | `op_tests/triton_tests/fusions/test_fused_kv_cache.py` |
| Triton fused BMM cache | `op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py` |
| Fused QK Norm + Cache | `op_tests/test_fused_qk_norm_rope_cache_quant.py` |
| PA with cache | `op_tests/test_pa.py` |
| MLA persistent | `op_tests/test_mla_persistent.py` |
| Batch prefill | `op_tests/test_batch_prefill.py` |
