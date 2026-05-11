---
title: "KV Cache Operators"
last_verified: 2026-04-06
source_files:
  - aiter/ops/cache.py
tags: [cache, kv-cache, paged-attention]
---

# KV Cache Operators

## Overview
aiter provides operators for managing paged KV caches used in autoregressive inference. These handle the physical page management, KV insertion, and format conversion needed by attention kernels.

## Key Functions

### Block Management
- `swap_blocks(src, dst, block_mapping)` -- swap KV blocks between GPU tensors (for beam search / preemption)
- `copy_blocks(key_caches, value_caches, block_mapping)` -- copy KV blocks (for forking sequences)

### KV Cache Write
- `reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale, asm_layout)` -- write new K/V tokens into paged cache with optional dtype conversion and quantization
- `reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)` -- flash-attention-compatible cache write
- `reshape_and_cache_with_pertoken_quant(key, value, key_cache, value_cache, k_dequant_scales, v_dequant_scales, slot_mapping, asm_layout)` -- cache write with per-token quantization

### Parameters
- `slot_mapping` -- maps each new token to its physical slot in the cache
- `kv_cache_dtype` -- target dtype for cached values (auto for same as input, or fp8)
- `k_scale`, `v_scale` -- quantization scales for FP8 cache
- `asm_layout` -- use assembly-optimized memory layout

## Cache Layout
The paged KV cache has shape `[num_pages, page_block_size, num_kv_heads, head_size]` for attention kernels that use the "flash" layout, or `[num_pages, num_kv_heads, page_block_size, head_size]` for the legacy PA decode layout. The `asm_layout` flag selects between these.

## Related Pages
- [[operators/attention]] -- attention kernels read from KV cache
- [[operators/rope]] -- RoPE can be fused with cache writes
- [[operators/quant]] -- quantization during cache insertion

## Source Files
- `aiter/ops/cache.py` -- all KV cache operations (~137 lines)
