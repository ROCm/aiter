---
title: "Attention Operators"
last_verified: 2026-04-06
source_files:
  - docs/attention_pipelines.md
  - docs/attention_kernels_explained.md
  - aiter/ops/attention.py
  - aiter/ops/mha.py
  - aiter/ops/unified_attention.py
  - aiter/ops/triton/attention/unified_attention.py
  - aiter/ops/triton/_triton_kernels/attention/unified_attention.py
  - aiter/ops/triton/_triton_kernels/attention/pa_decode.py
  - aiter/ops/triton/_triton_kernels/attention/pa_prefill.py
tags: [attention, triton, ck, paged-kv, gqa, flash-attention]
---

# Attention Operators

## Overview

Aiter provides 8 attention kernel variants across Triton and CK backends. All use the flash-attention pattern (online softmax, tiled computation, O(hdim) memory). The main entry points are `unified_attention()` for decode/mixed batches and `mha_varlen_fwd()` / `mha_varlen_fwd_pagedkv()` for CK flash-attention paths.

## Kernel Variants

| Kernel | Description |
|--------|-------------|
| **Triton 2D** | Single-pass unified attention with head-merge. Grid: `(num_kv_heads, total_num_q_blocks)`. Best for short KV or enough workgroups. |
| **Triton 3D** | Split-KV unified attention with head-merge. Two kernels (attention + reduce). Grid adds `NUM_SEGMENTS` dimension. Best for long KV, low batch. |
| **CK-UA** | CK Tile Unified Attention with GQA head-merge. Grid: `dim3(num_kv_heads, num_seqs)`. Best for decode with moderate batch (128–256 seqs on MI300X). |
| **CK-SK** | CK FMHA Split-KV. Two kernels (splitkv + combine). Best for decode, low batch + long KV. |
| **CK-PK** | CK FMHA PagedKV. Single kernel, no head-merge. Best for prefill with paged KV. |
| **CK-Fwd** | CK FMHA Forward (non-paged). Simplest path, contiguous KV. Rarely used in vLLM. |
| **Triton PA Decode** | Classic paged attention (legacy, being superseded by unified attention). |
| **Triton PA Prefill** | Context attention for prefill with existing KV cache. |

## Feature Matrix

| Feature | Triton 2D | Triton 3D | CK-UA | CK-SK | CK-PK | CK-Fwd | PA Decode | PA Prefill |
|---------|-----------|-----------|-------|-------|-------|--------|-----------|------------|
| Head-merge | Yes | Yes | Yes | Yes (decode) | No | No | No | No |
| Paged KV | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes |
| KV splitting | No | Yes | No | Yes | No | No | V2 only | No |
| Sliding window | Yes | Yes | No | Yes | Yes | Yes | No | No |
| Softcap | Yes | Yes | No | Yes | Yes | Yes | No | No |
| ALiBi | Yes | Yes | No | No | No | No | No | No |
| Sinks | Yes | Yes | No | Yes | Yes | Yes | No | No |
| FP8 output | Yes | No | No | No | No | No | No | No |
| Data types | fp16/bf16 | fp16/bf16 | fp16/bf16 | fp16/bf16/fp8 | fp16/bf16 | fp16/bf16/fp8 | fp16/bf16 | fp16/bf16 |
| Compiled hdim | Any | Any | 64/128 | 32–256 | 32–256 | 32–256 | Any | Any |

## Routing / Selector Logic

Dispatch hierarchy:

1. **`unified_attention()`** is the main entry point from vLLM.

2. **CK-UA first:** `_try_ck_unified_attention()` runs only for decode (`max_seqlen_q == 1`), with no sliding window, softcap, or ALiBi, for specific GQA configs (hdim 64 + GQA-8, or hdim 128 + MHA), and in a moderate occupancy band: `cu_count * 4 <= num_kv_heads * num_seqs <= cu_count * 8`. On MI300X with 8 KV heads, that is roughly 128–256 sequences.

3. **Triton 2D vs 3D:** If CK-UA does not apply, the path uses **2D** when sliding window is enabled, KV is short (≤ 512), or there are enough workgroups; otherwise **3D** (split-KV).

4. **`mha_varlen_fwd()`:** With `block_table` → **CK-SK**; without → **CK-Fwd**.

5. **`mha_varlen_fwd_pagedkv()`** → **CK-PK** directly.

## GQA Head-Merging

With GQA, several query heads share one KV head. **Head-merge** packs all Q heads in a GQA group into the M dimension of the Q tile so one MFMA can score all those heads at once. For decode with GQA-8, that yields 8 useful rows out of 16 (about 50% utilization) instead of roughly 0.8% without merging.

## CK-UA Tile Tiers

| Tier | kBlockM | Warps | MFMA | Use case |
|------|---------|-------|------|----------|
| Tiny | 16 | 1 | 16×16×32 | Pure decode (`avg_q` ≤ 2) |
| BS32 | 32 | 2 | 16×16×32 | `block_size=32` decode |
| Small | 64 | 2 | 32×32×16 | Short decode (`avg_q` ≤ 8) |
| Medium | 128 | 4 | 32×32×16 | All prefill |
| Large | 256 | 8 | 32×32×16 | Long prefill |

## Key Concepts

- **Paged KV cache:** `block_table[seq, page_col]` maps logical sequence positions to physical cache pages.
- **Online softmax:** Attention is computed in tiles with a running max and sum so memory stays O(head dim), not O(sequence length).
- **Split-KV reduce:** Partial attention results from parallel KV splits are merged with a log-sum-exp style reduction.

## Related Pages

- [[concepts/backend-selection]] — how aiter chooses between Triton and CK
- [[operators/cache]] — KV cache operations
- [[operators/mla]] — Multi-head Latent Attention (specialized attention variant)
- [[ck/attention-pipeline]] — CK unified attention pipeline internals

## Source Files

- `docs/attention_pipelines.md` — feature matrix, grid layouts, pseudocode for all 6 main kernels
- `docs/attention_kernels_explained.md` — detailed pseudocode with numerical examples
- `aiter/ops/attention.py` — Python entry point for attention ops
- `aiter/ops/mha.py` — CK flash attention wrappers (`mha_varlen_fwd`, `mha_fwd`)
- `aiter/ops/unified_attention.py` — CK-UA wrapper
- `aiter/ops/triton/attention/unified_attention.py` — Triton unified attention dispatcher + CK-UA selector
- `aiter/ops/triton/_triton_kernels/attention/unified_attention.py` — Triton 2D/3D kernel implementations
- `aiter/ops/triton/_triton_kernels/attention/pa_decode.py` — legacy paged attention decode
- `aiter/ops/triton/_triton_kernels/attention/pa_prefill.py` — prefill context attention
