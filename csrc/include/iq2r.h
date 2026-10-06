// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"

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
                                        int64_t expert_count);

// GLM-5.3 IQ2R packed MoE (6144 hidden, 256 routed + 1 fused shared expert,
// top-9). TP4 and TP8 are selected by the packed weight shapes.
enum Glm53GateKernel : int64_t
{
    kGlm53GateDecode           = 0,
    kGlm53GateDecodeNoBarrier  = 1,
    kGlm53GatePrefill          = 2,
};

enum Glm53DownKernel : int64_t
{
    kGlm53DownPacked  = 0,
    kGlm53DownOrdered = 1,
    kGlm53DownSingle  = 2,
};

void iq2r_glm53_sort_quant_out(const aiter_tensor_t& input,
                               const aiter_tensor_t& topk_ids,
                               aiter_tensor_t& sorted_expert_ids,
                               aiter_tensor_t& gather_indices,
                               aiter_tensor_t& scatter_indices,
                               aiter_tensor_t& tasks,
                               aiter_tensor_t& task_count,
                               aiter_tensor_t& output,
                               aiter_tensor_t& scales,
                               aiter_tensor_t& gate_tasks,
                               aiter_tensor_t& gate_task_count);

void iq2r_glm53_sort_out(const aiter_tensor_t& topk_ids,
                         aiter_tensor_t& sorted_expert_ids,
                         aiter_tensor_t& gather_indices,
                         aiter_tensor_t& scatter_indices,
                         aiter_tensor_t& tasks,
                         aiter_tensor_t& task_count,
                         aiter_tensor_t& scratch);

void iq2r_glm53_route_reduce_out(const aiter_tensor_t& route_output,
                                 const aiter_tensor_t& route_weights,
                                 const aiter_tensor_t& scatter_indices,
                                 aiter_tensor_t& output);

void iq2r_glm53_gate_m1_out(const aiter_tensor_t& activations,
                            const aiter_tensor_t& scales,
                            const aiter_tensor_t& data,
                            const aiter_tensor_t& auxiliary,
                            const aiter_tensor_t& tasks,
                            const aiter_tensor_t& task_count,
                            aiter_tensor_t& output,
                            aiter_tensor_t& output_scales);

void iq2r_glm53_gate_out(const aiter_tensor_t& activations,
                         const aiter_tensor_t& scales,
                         const aiter_tensor_t& data,
                         const aiter_tensor_t& auxiliary,
                         const aiter_tensor_t& tasks,
                         const aiter_tensor_t& task_count,
                         const aiter_tensor_t& gather,
                         aiter_tensor_t& output,
                         aiter_tensor_t& output_scales,
                         int64_t kernel,
                         int64_t grid_multiplier);

void iq2r_glm53_down_out(const aiter_tensor_t& activations,
                         const aiter_tensor_t& scales,
                         const aiter_tensor_t& data,
                         const aiter_tensor_t& auxiliary,
                         const aiter_tensor_t& tasks,
                         const aiter_tensor_t& task_count,
                         aiter_tensor_t& output,
                         int64_t kernel,
                         int64_t grid_multiplier);

void iq2r_glm53_down_route9_out(const aiter_tensor_t& activations,
                                const aiter_tensor_t& scales,
                                const aiter_tensor_t& data,
                                const aiter_tensor_t& auxiliary,
                                const aiter_tensor_t& expert_ids,
                                const aiter_tensor_t& scatter,
                                const aiter_tensor_t& route_weights,
                                aiter_tensor_t& output);

void iq2r_glm53_down_reduce_out(const aiter_tensor_t& activations,
                                const aiter_tensor_t& scales,
                                const aiter_tensor_t& data,
                                const aiter_tensor_t& auxiliary,
                                const aiter_tensor_t& tasks,
                                const aiter_tensor_t& task_count,
                                aiter_tensor_t& route_output,
                                const aiter_tensor_t& scatter,
                                const aiter_tensor_t& route_weights,
                                aiter_tensor_t& output,
                                int64_t chunks);

} // namespace aiter
