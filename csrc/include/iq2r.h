// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"

#include <optional>

namespace aiter {

void iq2r_encode_out(const aiter_tensor_t& weight,
                     const aiter_tensor_t& importance,
                     const aiter_tensor_t& codebook,
                     aiter_tensor_t& indices,
                     aiter_tensor_t& scales,
                     aiter_tensor_t& data,
                     aiter_tensor_t& auxiliary,
                     aiter_tensor_t& scale_delta_overflow,
                     int64_t valid_k,
                     int64_t exponent_radius,
                     double codebook_max);

void iq2r_materialize_out(const aiter_tensor_t& data,
                          const aiter_tensor_t& auxiliary,
                          aiter_tensor_t& output,
                          int64_t logical_n,
                          int64_t logical_k,
                          int64_t expert_index);

void iq2r_gemm_out(const aiter_tensor_t& activations,
                   const aiter_tensor_t& activation_scales,
                   const aiter_tensor_t& data,
                   const aiter_tensor_t& auxiliary,
                   std::optional<aiter_tensor_t> bias,
                   aiter_tensor_t& output,
                   int64_t logical_n,
                   int64_t logical_k,
                   int64_t tile_n,
                   int64_t expert_index);

void iq2r_task_gemm_out(const aiter_tensor_t& activations,
                        const aiter_tensor_t& activation_scales,
                        const aiter_tensor_t& data,
                        const aiter_tensor_t& auxiliary,
                        const aiter_tensor_t& tasks,
                        const aiter_tensor_t& task_count,
                        std::optional<aiter_tensor_t> bias,
                        aiter_tensor_t& output,
                        int64_t logical_n,
                        int64_t logical_k,
                        int64_t tile_n);

void iq2r_route_sort_tasks_out(const aiter_tensor_t& expert_ids,
                               aiter_tensor_t& sorted_expert_ids,
                               aiter_tensor_t& gather_indices,
                               aiter_tensor_t& scatter_indices,
                               aiter_tensor_t& tasks,
                               aiter_tensor_t& task_count,
                               int64_t expert_count,
                               int64_t expert_start,
                               int64_t task_rows);

void iq2r_route_gather_indexed_out(const aiter_tensor_t& input,
                                   const aiter_tensor_t& gather_indices,
                                   aiter_tensor_t& output,
                                   int64_t topk);

void iq2r_route_gather_quant_out(const aiter_tensor_t& input,
                                 const aiter_tensor_t& gather_indices,
                                 aiter_tensor_t& output,
                                 aiter_tensor_t& scales,
                                 int64_t topk);

void iq2r_route_direct_gather_quant_out(const aiter_tensor_t& input,
                                        const aiter_tensor_t& expert_ids,
                                        aiter_tensor_t& sorted_expert_ids,
                                        aiter_tensor_t& gather_indices,
                                        aiter_tensor_t& scatter_indices,
                                        aiter_tensor_t& tasks,
                                        aiter_tensor_t& task_count,
                                        aiter_tensor_t& output,
                                        aiter_tensor_t& scales,
                                        int64_t topk,
                                        int64_t expert_count,
                                        int64_t expert_start);

void iq2r_route_topk_direct_gather_quant_out(
    const aiter_tensor_t& input,
    const aiter_tensor_t& router_logits,
    aiter_tensor_t& topk_weights,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& sorted_expert_ids,
    aiter_tensor_t& gather_indices,
    aiter_tensor_t& scatter_indices,
    aiter_tensor_t& tasks,
    aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    aiter_tensor_t& scales,
    bool renormalize,
    std::optional<aiter_tensor_t> router_bias);

void iq2r_route_topk_sort_gather_quant_out(
    const aiter_tensor_t& input,
    const aiter_tensor_t& router_logits,
    aiter_tensor_t& topk_weights,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& sorted_expert_ids,
    aiter_tensor_t& gather_indices,
    aiter_tensor_t& scatter_indices,
    aiter_tensor_t& tasks,
    aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    aiter_tensor_t& scales,
    int64_t task_rows,
    bool renormalize,
    std::optional<aiter_tensor_t> router_bias);

void iq2r_swiglu_out(const aiter_tensor_t& gate_up,
                     aiter_tensor_t& output,
                     double limit,
                     double alpha,
                     double up_offset);

void iq2r_swiglu_quant_out(const aiter_tensor_t& gate_up,
                           aiter_tensor_t& output,
                           aiter_tensor_t& scales,
                           std::optional<aiter_tensor_t> activated,
                           double limit,
                           double alpha,
                           double up_offset);

void iq2r_route_reduce_indexed_out(const aiter_tensor_t& route_output,
                                   const aiter_tensor_t& route_weights,
                                   const aiter_tensor_t& scatter_indices,
                                   aiter_tensor_t& output,
                                   int64_t topk);

void iq2r_route_reduce_add_rmsnorm_indexed_out(
    const aiter_tensor_t& route_output,
    const aiter_tensor_t& route_weights,
    const aiter_tensor_t& scatter_indices,
    const aiter_tensor_t& residual,
    const aiter_tensor_t& norm_weight,
    aiter_tensor_t& output,
    aiter_tensor_t& residual_out,
    int64_t topk,
    double epsilon,
    int64_t block_size);

} // namespace aiter
