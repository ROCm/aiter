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

void iq2r_task_gemm_indexed_out(const aiter_tensor_t& activations,
                        const aiter_tensor_t& activation_scales,
                        const aiter_tensor_t& data,
                        const aiter_tensor_t& auxiliary,
                        const aiter_tensor_t& tasks,
                        const aiter_tensor_t& task_count,
                        std::optional<aiter_tensor_t> bias,
                        aiter_tensor_t& output,
                        int64_t logical_n,
                        int64_t logical_k,
                        int64_t tile_n,
                        const aiter_tensor_t& gather_indices);

void iq2r_task_gemm_swiglu_quant_out(const aiter_tensor_t& activations,
                                     const aiter_tensor_t& activation_scales,
                                     const aiter_tensor_t& data,
                                     const aiter_tensor_t& auxiliary,
                                     const aiter_tensor_t& tasks,
                                     const aiter_tensor_t& task_count,
                                     std::optional<aiter_tensor_t> bias,
                                     aiter_tensor_t& output,
                                     aiter_tensor_t& output_scales,
                                     int64_t logical_n,
                                     int64_t logical_k,
                                     double limit,
                                     double alpha,
                                     double up_offset);

void iq2r_route_sort_tasks_out(const aiter_tensor_t& expert_ids,
                               aiter_tensor_t& sorted_expert_ids,
                               aiter_tensor_t& gather_indices,
                               aiter_tensor_t& scatter_indices,
                               aiter_tensor_t& tasks,
                               aiter_tensor_t& task_count,
                               std::optional<aiter_tensor_t> expert_map,
                               int64_t expert_count,
                               int64_t expert_start,
                               int64_t expert_stride,
                               int64_t task_rows,
                               bool drop_nonlocal_tasks);

void iq2r_route_gather_indexed_out(const aiter_tensor_t& input,
                                   const aiter_tensor_t& gather_indices,
                                   aiter_tensor_t& output,
                                   int64_t topk);

void iq2r_route_gather_quant_out(const aiter_tensor_t& input,
                                 const aiter_tensor_t& gather_indices,
                                 aiter_tensor_t& output,
                                 aiter_tensor_t& scales,
                                 int64_t topk);

void iq2r_route_scatter_quant_out(const aiter_tensor_t& input,
                                  const aiter_tensor_t& scatter_indices,
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
                                        std::optional<aiter_tensor_t> expert_map,
                                        int64_t topk,
                                        int64_t expert_count,
                                        int64_t expert_start,
                                        int64_t expert_stride,
                                        bool drop_nonlocal_routes);

void iq2r_route_topk_direct_gather_quant_out(const aiter_tensor_t& input,
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
                                             std::optional<aiter_tensor_t> router_bias,
                                             bool biased_sigmoid,
                                             double routed_scaling_factor,
                                             std::optional<aiter_tensor_t> expert_map,
                                             int64_t expert_count,
                                             int64_t expert_start,
                                             int64_t expert_stride,
                                             bool drop_nonlocal_routes);

void iq2r_route_topk_sort_gather_quant_out(const aiter_tensor_t& input,
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
                                           std::optional<aiter_tensor_t> router_bias,
                                           bool biased_sigmoid,
                                           double routed_scaling_factor,
                                           std::optional<aiter_tensor_t> expert_map,
                                           int64_t expert_count,
                                           int64_t expert_start,
                                           int64_t expert_stride,
                                           bool drop_nonlocal_routes);

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

void iq2r_swiglu_quant_scatter_out(const aiter_tensor_t& gate_up,
                                   const aiter_tensor_t& scatter_indices,
                                   aiter_tensor_t& output,
                                   aiter_tensor_t& scales,
                                   int64_t topk,
                                   std::optional<aiter_tensor_t> activated,
                                   double limit,
                                   double alpha,
                                   double up_offset);

void iq2r_route_reduce_indexed_out(const aiter_tensor_t& route_output,
                                   const aiter_tensor_t& route_weights,
                                   const aiter_tensor_t& scatter_indices,
                                   aiter_tensor_t& output,
                                   int64_t topk);

void iq2r_route_reduce_add_indexed_out(const aiter_tensor_t& route_output,
                                       const aiter_tensor_t& route_weights,
                                       const aiter_tensor_t& scatter_indices,
                                       const aiter_tensor_t& shared_output,
                                       aiter_tensor_t& output,
                                       int64_t topk);

void iq2r_route_reduce_add_rmsnorm_indexed_out(const aiter_tensor_t& route_output,
                                               const aiter_tensor_t& route_weights,
                                               const aiter_tensor_t& scatter_indices,
                                               const aiter_tensor_t& residual,
                                               const aiter_tensor_t& norm_weight,
                                               aiter_tensor_t& output,
                                               aiter_tensor_t& residual_out,
                                               int64_t topk,
                                               double epsilon,
                                               int64_t block_size);

void iq2r_gate_aligned_fused_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather,
    aiter_tensor_t& output,
    aiter_tensor_t& output_scales,
    int64_t rows_per_cta);
void iq2r_gate_quad_fused_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather,
    aiter_tensor_t& output,
    aiter_tensor_t& output_scales,
    int64_t rows_per_cta);
void iq2r_gate_quad_sparse_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather,
    aiter_tensor_t& output,
    aiter_tensor_t& output_scales,
    int64_t rows_per_cta);
void iq2r_gate_quad_splitk_out(
    const aiter_tensor_t& input,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    aiter_tensor_t& partials,
    aiter_tensor_t& output,
    aiter_tensor_t& output_scales,
    int64_t physical_waves);
void iq2r_gate_quad_route_fused_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    aiter_tensor_t& output_scales);
void iq2r_down_sparse_large32_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    int64_t grid_multiplier);
void iq2r_down_sparse_scheduled_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    int64_t grid_multiplier,
    int64_t variant);
void iq2r_down_shortk_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    int64_t grid_multiplier,
    int64_t variant);
void iq2r_gate_quad_scheduled_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    aiter_tensor_t& output_scales,
    int64_t variant);
void iq2r_down_token_fused48_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids,
    const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights,
    aiter_tensor_t& output);
void iq2r_down_token_route9_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids,
    const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights,
    aiter_tensor_t& output);
void iq2r_glm53_tp4_gate_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    aiter_tensor_t& output_scales,
    int64_t variant);
void iq2r_glm53_tp4_down_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    int64_t grid_multiplier,
    int64_t variant);
void iq2r_glm53_tp4_route9_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids,
    const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights,
    aiter_tensor_t& output);
void iq2r_glm53_tp4_indexed_gate_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather,
    aiter_tensor_t& output,
    aiter_tensor_t& output_scales,
    int64_t rows_per_cta);
void iq2r_glm53_tp4_large_down_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    aiter_tensor_t& output,
    int64_t grid_multiplier,
    int64_t variant);
void iq2r_down_token_pair9_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids,
    const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights,
    aiter_tensor_t& output,
    int64_t group_tokens);
void iq2r_glm53_dense_gate_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& tasks,
    const aiter_tensor_t& task_count,
    const aiter_tensor_t& gather,
    aiter_tensor_t& output,
    aiter_tensor_t& output_scales,
    int64_t rows_per_cta,
    int64_t variant);
void iq2r_down_token_adaptive9_out(
    const aiter_tensor_t& activations,
    const aiter_tensor_t& scales,
    const aiter_tensor_t& data,
    const aiter_tensor_t& auxiliary,
    const aiter_tensor_t& expert_ids,
    const aiter_tensor_t& scatter,
    const aiter_tensor_t& route_weights,
    aiter_tensor_t& output,
    const aiter_tensor_t& task_count,
    const aiter_tensor_t& task_table);
} // namespace aiter
