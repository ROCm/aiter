#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MiniMax-M3.1 TP1 decoder-layer kernels for gfx950 (aiter/ops/m3_tp1_prefill.py).
// Torch-free: tensors cross as aiter_tensor_t, launches go to aiter::getCurrentHIPStream().
#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

// Gemma add+RMSNorm (fused_qk_rmsnorm_group_quant, BlockSize 512) with MX epilogues.
void m3_tp1_norm_mx(const aiter_tensor_t& x, // [m, n] bf16, [m, 4, n] with RED4
                    int64_t flags,           // a NORM_* set of aiter/ops/m3_tp1_prefill.py
                    std::optional<aiter_tensor_t> s_add,    // [m, n] bf16 (ADD3)
                    std::optional<aiter_tensor_t> res,      // [m, n] bf16
                    std::optional<aiter_tensor_t> w,        // [n] bf16
                    std::optional<aiter_tensor_t> res_out,  // [m, n] bf16
                    std::optional<aiter_tensor_t> out_bf16, // [m, n] bf16 (OUT_BF16, HOUT)
                    std::optional<aiter_tensor_t> q8,       // [m, n] e4m3 (MX8)
                    std::optional<aiter_tensor_t> s8,       // [>= m, n / 32] e8m0, or w4 (MX8)
                    std::optional<aiter_tensor_t> q4,       // [m, n / 2] e2m1 x2 (MX4)
                    std::optional<aiter_tensor_t> s4,       // [m, n / 32] e8m0 (MX4)
                    double eps,
                    int64_t m,
                    int64_t n,
                    bool s8_w4);

// bf16 [rows, K] -> e4m3 [rows, K] + e8m0 scales, row-major [rows, K / 32] or shuffle_scale_w4.
void m3_tp1_mx8_quant(const aiter_tensor_t& x, aiter_tensor_t& q, aiter_tensor_t& s, bool w4);

// MoE GEMM2 (H 6144, I 3072, sort block 128) on ragged 256-row expert tiles: the per-route bf16
// rows out[token, slot] of aiter's v2 "reduce" GEMM2. cnt / tiles / ntiles: tile-list scratch.
void m3_tp1_moe_g2(const aiter_tensor_t& a2,                // [sorted, 1536] MXFP4, GEMM1's output
                   const aiter_tensor_t& a2_scale,          // its sorted, shuffled e8m0 scales
                   const aiter_tensor_t& w2_b,              // relayout_w2 weight bytes
                   const aiter_tensor_t& w2_bs,             // relayout_w2 scale bytes
                   const aiter_tensor_t& sorted_ids,        // [sorted] int32, token | slot << 24
                   const aiter_tensor_t& sorted_weights,    // [sorted] fp32
                   const aiter_tensor_t& sorted_expert_ids, // [sorted / 128] int32
                   const aiter_tensor_t& num_valid_ids,     // [>= 1] int32
                   aiter_tensor_t& cnt,                     // [>= sorted / 128] int32
                   aiter_tensor_t& tiles,                   // [>= sorted / 128, 4] int32
                   aiter_tensor_t& ntiles,                  // [>= 1] int32
                   aiter_tensor_t& out,                     // [m, topk, 6144] bf16
                   int64_t m,
                   int64_t topk);

} // namespace aiter
