---
title: "CK Unified Attention Pipeline"
last_verified: 2026-04-06
source_files:
  - 3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp
  - 3rdparty/composable_kernel/example/ck_tile/42_unified_attention/
tags: [ck, attention, unified-attention, pipeline, mfma]
---

# CK Unified Attention Pipeline

## Overview
The CK Unified Attention Pipeline (`UnifiedAttentionPipeline`) is the core CK-Tile implementation of paged-KV attention with GQA head-merging. This is the kernel behind CK-UA as described in [[operators/attention]].

## Template Structure
The pipeline is parameterized by:
- `Problem_` -- defines data types (Q/K/V/O), accumulator type, shapes, and mask behavior
- `Policy_` -- defines tile sizes (kBlockM, kBlockQ), warp counts, page block size, and MFMA instruction selection

Key compile-time parameters from the policy:
- `kBlockM` -- Q tile height (16, 32, 64, 128, 256 depending on tier)
- `kBlockQ` -- Q tokens per tile = kBlockM / num_queries_per_kv
- `kPageBlockSize` -- KV page size (32 or 64)
- `kHeadDim` / `kHeadDimPadded` -- head dimension (64 or 128, padded to alignment)

## Tile Tiers
Selection based on `avg_q = num_tokens / num_seqs`:

| Tier | kBlockM | Warps | MFMA | Grid | Use Case |
|------|---------|-------|------|------|----------|
| Tiny | 16 | 1 | 16x16x32 | 2D decode | Pure decode (avg_q <= 2) |
| BS32 | 32 | 2 | 16x16x32 | 2D decode | block_size=32 decode |
| Small | 64 | 2 | 32x32x16 | 2D decode | Short decode (avg_q <= 8) |
| Medium | 128 | 4 | 32x32x16 | 1D prefill | All prefill |
| Large | 256 | 8 | 32x32x16 | 1D prefill | Long prefill |

## GQA Head-Merging
The key optimization: all Q heads sharing a KV head are packed into the M dimension of a single workgroup. With GQA-8 and kBlockM=16: 2 tokens x 8 heads = 16 rows, one MFMA computes scores for all 8 heads simultaneously.

Grid for decode: `dim3(num_kv_heads, num_seqs)` -- only 32 workgroups for batch=4 with 8 KV-heads, but each does 8x more useful work than CK-PK's 256 workgroups.

## Pipeline Flow
1. Load Q tile with GQA merge
2. For each KV page: async load K/V from paged cache -> MFMA QK^T -> causal mask -> online softmax -> MFMA PV accumulate
3. Un-merge heads and write output

## Debug/Profile Hooks
The pipeline includes:
- `ASM_MARKER` for ROCm profiler markers
- Optional debug predicates on block/warp/lane indices
- Static assertions validating dtype and head-dim constraints (kHeadDimPadded <= 256)

## Related Pages
- [[operators/attention]] -- high-level attention operator documentation
- [[ck/architecture]] -- CK's four-layer model and MFMA instructions
- [[concepts/backend-selection]] -- CK-UA selector logic

## Source Files
- `3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp` -- main pipeline (~1151 lines)
- `3rdparty/composable_kernel/example/ck_tile/42_unified_attention/` -- example and test code
