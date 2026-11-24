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
// Fixed at n_heads=32, head_dim=128 -- the GLM-5.x indexer shape.
//
// No torch dependency: tensors cross as aiter_tensor_t and the stream comes from
// aiter::getCurrentHIPStream(). The output is allocated by the Python wrapper
// (aiter/ops/fp8_mqa_logits.py).
#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

// Writes logits into `out` ([M, N] f32, contiguous). Positions outside a row's
// [ks, ke) window are written -inf when `clean_logits`, and left untouched otherwise.
void fp8_mqa_logits(aiter_tensor_t& out,                // [M, N]       f32
                    const aiter_tensor_t& q_fp8,        // [M, 32, 128] fp8
                    const aiter_tensor_t& k_fp8,        // [N, 128]     fp8
                    const aiter_tensor_t& kv_scale,     // [N]          f32
                    const aiter_tensor_t& weights,      // [M, 32]      f32
                    const aiter_tensor_t& cu_seqlen_ks, // [M]          i32
                    const aiter_tensor_t& cu_seqlen_ke, // [M]          i32
                    int64_t BlockM,
                    int64_t SplitN,
                    int64_t num_warps,
                    int64_t TotalCuCount,
                    bool clean_logits,
                    int64_t ReverseRows); // -1 = host heuristic, 0/1 = off/on

} // namespace aiter
