#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Hand-written HIP prefill-phase FP8 MQA indexer logits kernel for gfx950.
//
// The dense half of the DeepSeek-V3.2 / GLM-5 sparse-attention lightning indexer:
//
//   logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]
//                  for n in [ks[m], ke[m]), -inf elsewhere
//
// K has already been gathered out of the paged cache into a contiguous [N, 128]
// buffer, so there is no block table. Same contract as
// `aiter.ops.triton.attention.fp8_mqa_logits`, so it drops into the same call site.
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

// Writes into `out` when given (and returns it), otherwise allocates [M, N] f32.
// Positions outside a row's [ks, ke) window read -inf when `clean_logits`, and are
// left untouched otherwise.
torch::Tensor fp8_mqa_logits(torch::Tensor q_fp8,        // [M, 32, 128] fp8
                             torch::Tensor k_fp8,        // [N, 128]     fp8
                             torch::Tensor kv_scale,     // [N]          f32
                             torch::Tensor weights,      // [M, 32]      f32
                             torch::Tensor cu_seqlen_ks, // [M]          i32
                             torch::Tensor cu_seqlen_ke, // [M]          i32
                             int64_t BlockM,
                             int64_t SplitN,
                             int64_t num_warps,
                             int64_t TotalCuCount,
                             bool clean_logits,
                             int64_t Unroll2,     // -1 = host heuristic, 0/1 = off/on
                             int64_t ReverseRows, // -1 = host heuristic, 0/1 = off/on
                             std::optional<torch::Tensor> out);
