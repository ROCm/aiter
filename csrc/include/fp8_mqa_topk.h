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
//
// No torch dependency: tensors cross as aiter_tensor_t, the stream comes from
// aiter::getCurrentHIPStream(), and all scratch memory is one caller-allocated workspace of
// fp8_mqa_topk_workspace_size() bytes (allocated by aiter/ops/fp8_mqa_topk.py).
#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

constexpr int FP8_MQA_TOPK_MAX_K = 512;

namespace aiter {

// Bytes of workspace fp8_mqa_topk() needs for M query rows and N keys (0 when M or N is 0).
int64_t fp8_mqa_topk_workspace_size(int64_t M, int64_t N, int64_t TotalCuCount);

// Top-K per row (K = topk_indices.size(1) <= 512), written into topk_indices: row m gets the
// indices, relative to its first valid key max(ks[m], 0), of its K largest logits in
// [ks[m], ke[m]) in no particular order, or 0..len-1 then -1 for a row with len <= K
// (top_k_per_row_prefill writes the same sets as absolute columns). The logits are bit-identical to
// fp8_mqa_logits', so the selection is a top-K of that kernel's output (exact ties at the K-th
// value broken arbitrarily). workspace: uninitialised, at least
// fp8_mqa_topk_workspace_size(M, N, TotalCuCount) bytes. fallback_rows (optional, [1] int32):
// incremented by the number of rows that took the exact fallback. force_fallback (test hook): no
// candidates, so every row of more than K keys takes the exact fallback.
void fp8_mqa_topk(const aiter_tensor_t& q_fp8,        // [M, 32, 128] fp8
                  const aiter_tensor_t& k_fp8,        // [N, 128]     fp8
                  const aiter_tensor_t& kv_scale,     // [N]          f32
                  const aiter_tensor_t& weights,      // [M, 32]      f32
                  const aiter_tensor_t& cu_seqlen_ks, // [M]          i32
                  const aiter_tensor_t& cu_seqlen_ke, // [M]          i32
                  aiter_tensor_t& topk_indices,       // [M, K]       i32
                  aiter_tensor_t& workspace,          // [bytes]      u8
                  int64_t TotalCuCount,
                  std::optional<aiter_tensor_t> fallback_rows,
                  bool force_fallback);

} // namespace aiter
