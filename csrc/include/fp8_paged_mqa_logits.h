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

// aiter builds csrc without hipify, so the ATen CUDA headers (which pull in
// cuda_runtime_api.h) are unavailable -- use the HIP-flavoured ones, as the rest
// of csrc does.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <cstdlib>
#include <type_traits>

// Writes into `out` when given (and returns it), otherwise allocates
// [batch*next_n, max_model_len] f32. Only the causal window
// p <= context_lens[b] - next_n + n is written; the -inf outside it is the
// caller's, exactly as for deepgemm_fp8_paged_mqa_logits.
torch::Tensor
fp8_paged_mqa_logits(torch::Tensor q_fp8,        // [batch, next_n, 32, 128] fp8
                     torch::Tensor kv_cache_fp8, // [blocks, bsz, 1, index_dim];
                                                 // bsz=1 plain, bsz=64 preshuffled
                     torch::Tensor weights,      // [batch*next_n, 32] f32
                     torch::Tensor context_lens, // [batch]             i32
                     torch::Tensor block_tables, // [batch, max_blocks] i32
                     int64_t max_model_len,
                     int64_t ChunkK,
                     int64_t SplitKV,
                     int64_t num_warps,
                     int64_t TotalCuCount,
                     int64_t RowsPerBlock,
                     std::optional<torch::Tensor> out);
