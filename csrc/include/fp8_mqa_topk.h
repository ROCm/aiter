#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Hand-written HIP prefill-phase FP8 MQA indexer logits + per-row top-K for gfx950, fused so the
// fp32 [M, N] logits are never written:
//
//   logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]   for n in [ks[m], ke[m])
//
// K has already been gathered into a contiguous [N, 128] buffer (same inputs as
// aiter.ops.fp8_mqa_logits). Fixed at n_heads=32, head_dim=128 -- the GLM-5.x indexer shape.
// Pipeline: csrc/kernels/fp8_mqa_topk.cu.

// aiter builds csrc without hipify, so the ATen CUDA headers (which pull in
// cuda_runtime_api.h) are unavailable -- use the HIP-flavoured ones, as the rest
// of csrc does.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <torch/extension.h>

constexpr int FP8_MQA_TOPK_MAX_K = 512;

// Top-K per row (K = topk_indices.size(1) <= 512), written into topk_indices (and returned): row m
// gets the indices, relative to its first valid key max(ks[m], 0), of its K largest logits in
// [ks[m], ke[m]) in no particular order, or 0..len-1 then -1 for a row with len <= K
// (top_k_per_row_prefill writes the same sets as absolute columns). The logits are bit-identical to
// fp8_mqa_logits', so the selection is a top-K of that kernel's output (exact ties at the K-th
// value broken arbitrarily). fallback_rows (optional, [1] int32): incremented by the number of rows
// that took the exact fallback. force_fallback (test hook): no candidates, so every row of more
// than K keys takes the exact fallback.
torch::Tensor fp8_mqa_topk(torch::Tensor q_fp8,        // [M, 32, 128] fp8
                           torch::Tensor k_fp8,        // [N, 128]     fp8
                           torch::Tensor kv_scale,     // [N]          fp32
                           torch::Tensor weights,      // [M, 32]      fp32
                           torch::Tensor cu_seqlen_ks, // [M]          int32
                           torch::Tensor cu_seqlen_ke, // [M]          int32
                           torch::Tensor topk_indices, // [M, K]       int32, written in place
                           int64_t TotalCuCount,
                           std::optional<torch::Tensor> fallback_rows,
                           bool force_fallback);
