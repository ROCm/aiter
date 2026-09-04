// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

// Distribution-independent upper bound on the rows emitted by MoE sorting.
// With R route slots and a non-empty experts, writing each positive row count
// as c_i = 1 + x_i gives
//
//   sum(ceil(c_i / block_m)) = a + sum(floor(x_i / block_m))
//                           <= a + floor((R - a) / block_m),
//
// which is non-decreasing in both a and R. So a = min(num_experts, R) is the
// worst case, and the bound still holds when routes are dropped as invalid.
constexpr int64_t moe_quant_padded_rows_upper_bound(int64_t total_routes,
                                                    int64_t num_experts,
                                                    int64_t block_m) noexcept
{
    if(total_routes <= 0 || num_experts <= 0 || block_m <= 0)
    {
        return 0;
    }
    const int64_t active_experts = num_experts < total_routes ? num_experts : total_routes;
    return block_m * (active_experts + (total_routes - active_experts) / block_m);
}

static_assert(moe_quant_padded_rows_upper_bound(1, 7, 32) == 32);
static_assert(moe_quant_padded_rows_upper_bound(8, 7, 32) == 224);
static_assert(moe_quant_padded_rows_upper_bound(504, 7, 32) == 704);
static_assert(moe_quant_padded_rows_upper_bound(1024, 2, 32) == 1056);
static_assert(moe_quant_padded_rows_upper_bound(64, 64, 64) == 4096);

void static_per_tensor_quant(aiter_tensor_t& out,          // [..., d]
                             const aiter_tensor_t& input,  // [..., d]
                             const aiter_tensor_t& scale); // [1]

void dynamic_per_tensor_quant(aiter_tensor_t& out,         // [..., d]
                              const aiter_tensor_t& input,  // [..., d]
                              aiter_tensor_t& scale);       // [1]

void dynamic_per_token_scaled_quant(aiter_tensor_t& out,         // [..., d]
                                    const aiter_tensor_t& input, // [..., d]
                                    aiter_tensor_t& scales,
                                    std::optional<aiter_tensor_t> scale_ub  = std::nullopt,
                                    bool shuffle_scale                      = false,
                                    std::optional<aiter_tensor_t> num_rows  = std::nullopt,
                                    int num_rows_factor                     = 1);

// Canonical dtype-aware per-group dynamic quant. Accepts fp8 / i8 / fp4x2.
// For fp4x2 it writes an e8m0 byte per group; for fp8/i8 it writes an
// fp32 per-group scale.
void dynamic_per_group_scaled_quant(aiter_tensor_t& out,         // [..., d]
                                    const aiter_tensor_t& input, // [..., d]
                                    aiter_tensor_t& scales,
                                    int group_size                             = 32,
                                    bool shuffle_scale                         = true,
                                    std::optional<aiter_tensor_t> num_rows     = std::nullopt,
                                    int num_rows_factor                        = 1);

// Backward-compat fp4-only entry; delegates to dynamic_per_group_scaled_quant.
void dynamic_per_group_scaled_quant_fp4(aiter_tensor_t& out,         // [..., d]
                                        const aiter_tensor_t& input, // [..., d]
                                        aiter_tensor_t& scales,
                                        int group_size                             = 32,
                                        bool shuffle_scale                         = true,
                                        std::optional<aiter_tensor_t> num_rows     = std::nullopt,
                                        int num_rows_factor                        = 1);

// Private pybind target for the direct-M1 executor; not part of the public quant API.
void dynamic_per_group_scaled_quant_fp4_direct_m1_internal(
    aiter_tensor_t& out,
    const aiter_tensor_t& input,
    aiter_tensor_t& scales,
    aiter_tensor_t& zero_output);

void smooth_per_token_scaled_quant(
    aiter_tensor_t& out,         // [..., d]
    const aiter_tensor_t& input, // [..., d]
    aiter_tensor_t& scales,
    const aiter_tensor_t& smooth_scale,
    std::optional<aiter_tensor_t> smooth_scale_map      = std::nullopt,
    bool shuffle_scale                                  = false,
    std::optional<aiter_tensor_t> num_rows              = std::nullopt,
    int num_rows_factor                                 = 1,
    std::optional<aiter_tensor_t> smooth_scale_map_hash = std::nullopt,
    bool enable_ps                                      = true);

void partial_transpose(aiter_tensor_t& out,         // [rows, d]
                       const aiter_tensor_t& input, // [rows, d]
                       const aiter_tensor_t& num_rows);

void moe_smooth_per_token_scaled_quant_v1(
    aiter_tensor_t& out,         // [..., d]
    const aiter_tensor_t& input, // [..., d]
    aiter_tensor_t& scales,
    const aiter_tensor_t& smooth_scale,
    const aiter_tensor_t& smooth_scale_map,
    bool shuffle_scale                                  = false,
    std::optional<aiter_tensor_t> smooth_scale_map_hash = std::nullopt,
    bool transpose_out                                  = false);

void moe_smooth_per_token_scaled_quant_v2(aiter_tensor_t& out,         // [..., d]
                                          const aiter_tensor_t& input, // [..., d]
                                          aiter_tensor_t& scales,
                                          const aiter_tensor_t& smooth_scale,
                                          const aiter_tensor_t& sorted_token_ids,
                                          const aiter_tensor_t& sorted_expert_ids,
                                          const aiter_tensor_t& num_valid_ids,
                                          int block_m,
                                          bool shuffle_scale = false,
                                          bool transpose_out = false);

void fused_dynamic_mx_quant_moe_sort_hip(aiter_tensor_t& out,         // [token_num * topk, d] for fp8 or [token_num * topk, d / 2] for fp4
                                            aiter_tensor_t& scales,      // swizzled e8m0 bytes
                                            const aiter_tensor_t& input, // [token_num * topk, d]
                                            const aiter_tensor_t& sorted_ids,
                                            const aiter_tensor_t& num_valid_ids,
                                            int token_num,
                                            int block_m,
                                            int group_size = 32,
                                            std::optional<aiter_tensor_t> sorted_weights = std::nullopt);

void fused_dynamic_mx_quant_moe_sort_hip_bounded(
    aiter_tensor_t& out,         // [token_num * topk, d] for fp8 or [token_num * topk, d / 2] for fp4
    aiter_tensor_t& scales,      // swizzled e8m0 bytes
    const aiter_tensor_t& input, // [token_num * topk, d]
    const aiter_tensor_t& sorted_ids,
    const aiter_tensor_t& num_valid_ids,
    int token_num,
    int block_m,
    int64_t total_routes,
    int64_t num_experts_upper_bound,
    int group_size                                      = 32,
    std::optional<aiter_tensor_t> sorted_weights        = std::nullopt);

void mxfp4_moe_sort_hip(aiter_tensor_t& out_scale,
                         const aiter_tensor_t& scale,
                         const aiter_tensor_t& sorted_ids,
                         const aiter_tensor_t& num_valid_ids,
                         int token_num,
                         int cols);
void quant_mxfp4(const aiter_tensor_t& inp,
                 aiter_tensor_t& out_packed,
                 aiter_tensor_t& out_scale,
                 int group_size          = 32,
                 int round_mode          = 0,
                 bool e8m0_shuffle       = false,
                 bool a16w4_shuffle      = false,
                 bool gate_up            = false,
                 bool shuffle_weight     = false);
void quant_mxfp6_gemm_hip(const aiter_tensor_t& input,
                          aiter_tensor_t& packed,
                          aiter_tensor_t& packed_scale);
} // namespace aiter
