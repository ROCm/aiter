#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Hand-written HIP decode-phase FP8 paged MQA indexer logits kernel for gfx950.
//
// The paged half of the DeepSeek-V3.2 / GLM-5 sparse-attention lightning indexer:
//
//   logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]
//
// K and its per-token dequant scale are co-packed in a paged cache and gathered
// through a block table. Same tensor contract as
// `deepgemm_fp8_paged_mqa_logits`, so it drops into the same call site.
//
// Fixed at n_heads=32, head_dim=128 -- the shipped GLM-5-FP8 indexer shape.
//
// No torch dependency: tensors cross as aiter_tensor_t and the stream comes from
// aiter::getCurrentHIPStream(). The output is allocated by the Python wrapper
// (aiter/ops/fp8_paged_mqa_logits.py) when the caller passes none.
#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

// Writes into `out` ([batch*next_n, max_model_len] f32, contiguous). Only the causal
// window p <= context_lens[b] - next_n + n is written ([batch] lengths), or
// p < context_lens[b][n] ([batch, next_n] lengths, one per query row); the -inf
// outside it is the caller's, exactly as for deepgemm_fp8_paged_mqa_logits.
void fp8_paged_mqa_logits(aiter_tensor_t& out,                // [batch*next_n, max_model_len] f32
                          const aiter_tensor_t& q_fp8,        // [batch, next_n, 32, 128] fp8
                          const aiter_tensor_t& kv_cache_fp8, // [blocks, bsz, 1, index_dim];
                                                              // bsz=1 plain, bsz=64 preshuffled
                          const aiter_tensor_t& weights,      // [batch*next_n, 32] f32
                          const aiter_tensor_t& context_lens, // [batch] or [batch, next_n] i32
                          const aiter_tensor_t& block_tables, // [batch, max_blocks] i32
                          int64_t max_model_len,
                          int64_t ChunkK,
                          int64_t SplitKV,
                          int64_t num_warps,
                          int64_t TotalCuCount,
                          int64_t RowsPerBlock);

} // namespace aiter
