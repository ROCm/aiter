// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"

namespace aiter {

// Raises unless the current GPU is gfx950; the IQ2R kernels are built for gfx950 only.
void iq2r_check_gfx950();

// GLM-5.3 IQ2R-v2 packed MoE (6144 hidden, 256 routed experts, top-8). TP4
// and TP8 are selected by the packed weight shapes. The decode paths (M <= 1024)
// can run the MXFP8 shared expert as route slot 8; empty shared tensors leave
// it out.
//
// Gate kernels of iq2r_glm53_gate_out (aiter/iq2r_glm53.py GATE_KERNELS):
enum Glm53GateKernel : int64_t
{
    // "decode": M2..1024, M16 tasks; each wave sums a K768 slice.
    kGlm53GateDecode = 1,
    // "prefill": M2..4096, M64 tasks sharing a K128 activation tile.
    kGlm53GatePrefill = 2,
};

// Down kernels of iq2r_glm53_down_out (aiter/iq2r_glm53.py DOWN_KERNELS). The
// per-token down ("per_token", M1/M2/M4) has its own entry point,
// iq2r_glm53_down_per_token_out.
enum Glm53DownKernel : int64_t
{
    // "m32": TP8, two 16-token M atoms per pass.
    kGlm53DownM32 = 1,
    // "m16": one 16-token M atom per pass; at TP4, three workgroups per CU.
    kGlm53DownM16 = 2,
    // "m16_2wg": the TP4 m16 built for two workgroups per CU (more VGPRs per wave).
    kGlm53DownM16TwoWg = 3,
};

void iq2r_glm53_m1_route_quant_out(const aiter_tensor_t& input,
                                   const aiter_tensor_t& topk_ids,
                                   aiter_tensor_t& scatter_indices,
                                   aiter_tensor_t& tasks,
                                   aiter_tensor_t& task_count,
                                   aiter_tensor_t& output,
                                   aiter_tensor_t& scales,
                                   int64_t shared);

void iq2r_glm53_sort_quant_out(const aiter_tensor_t& input,
                               const aiter_tensor_t& topk_ids,
                               aiter_tensor_t& gather_indices,
                               aiter_tensor_t& scatter_indices,
                               aiter_tensor_t& tasks,
                               aiter_tensor_t& task_count,
                               aiter_tensor_t& output,
                               aiter_tensor_t& scales,
                               aiter_tensor_t& gate_tasks,
                               aiter_tensor_t& gate_task_count,
                               int64_t shared);

void iq2r_glm53_prefill_sort_quant_out(const aiter_tensor_t& input,
                                       const aiter_tensor_t& topk_ids,
                                       aiter_tensor_t& gather_indices,
                                       aiter_tensor_t& scatter_indices,
                                       aiter_tensor_t& tasks,
                                       aiter_tensor_t& task_count,
                                       aiter_tensor_t& output,
                                       aiter_tensor_t& scales,
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
                            aiter_tensor_t& output_scales,
                            const aiter_tensor_t& shared_gate,
                            const aiter_tensor_t& shared_gate_scales,
                            aiter_tensor_t& shared_act);

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
                         int64_t grid_multiplier,
                         const aiter_tensor_t& shared_gate,
                         const aiter_tensor_t& shared_gate_scales,
                         aiter_tensor_t& shared_act);

void iq2r_glm53_down_out(const aiter_tensor_t& activations,
                         const aiter_tensor_t& scales,
                         const aiter_tensor_t& data,
                         const aiter_tensor_t& auxiliary,
                         const aiter_tensor_t& tasks,
                         const aiter_tensor_t& task_count,
                         aiter_tensor_t& output,
                         int64_t kernel,
                         int64_t grid_multiplier,
                         const aiter_tensor_t& scatter,
                         const aiter_tensor_t& shared_down,
                         const aiter_tensor_t& shared_down_scales,
                         const aiter_tensor_t& shared_act);

void iq2r_glm53_down_per_token_out(const aiter_tensor_t& activations,
                                   const aiter_tensor_t& scales,
                                   const aiter_tensor_t& data,
                                   const aiter_tensor_t& auxiliary,
                                   const aiter_tensor_t& expert_ids,
                                   const aiter_tensor_t& scatter,
                                   const aiter_tensor_t& route_weights,
                                   aiter_tensor_t& output,
                                   const aiter_tensor_t& shared_down,
                                   const aiter_tensor_t& shared_down_scales,
                                   const aiter_tensor_t& shared_act);

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
