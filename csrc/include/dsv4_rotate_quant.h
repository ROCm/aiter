// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

void rotate_activation_fp4quant(aiter_tensor_t& out,
                                        aiter_tensor_t& scale,
                                        const aiter_tensor_t& input,
                                        int32_t group_size = 32,
                                        bool shuffle_scale = true);


void rotate_activation(aiter_tensor_t& out,
                       const aiter_tensor_t& input);

void rope_rotate_activation_fp4quant(aiter_tensor_t& out,
                                            aiter_tensor_t& scale,
                                            const aiter_tensor_t& input,
                                            const aiter_tensor_t& cos,
                                            const aiter_tensor_t& sin,
                                            const aiter_tensor_t& positions,
                                            int32_t rope_dim,
                                            int32_t group_size = 32,
                                            bool shuffle_scale = true);

// rope+hadamard, bf16/fp16 in-place (out shares dtype/stride with input).
void rope_rotate_activation(aiter_tensor_t& out,
                            const aiter_tensor_t& input,
                            const aiter_tensor_t& cos,
                            const aiter_tensor_t& sin,
                            const aiter_tensor_t& positions,
                            int32_t rope_dim);

// rope+hadamard, then fp8-quantize: `out` is fp8 and `scale` receives
// per-(row, 1xGROUP) fp32 scales ([m, dim/group_size]), matching
// get_hip_quant(per_1x128). Symmetric to rope_rotate_activation_fp4quant.
void rope_rotate_activation_fp8quant(aiter_tensor_t& out,
                                     aiter_tensor_t& scale,
                                     const aiter_tensor_t& input,
                                     const aiter_tensor_t& cos,
                                     const aiter_tensor_t& sin,
                                     const aiter_tensor_t& positions,
                                     int32_t rope_dim,
                                     int32_t group_size = 128);

void rmsnorm_rope_rotate_activation_fp4quant_kvcache(aiter_tensor_t& kvcache,
                                                     aiter_tensor_t& scale,
                                                     const aiter_tensor_t& input,
                                                     const aiter_tensor_t& norm_weight,
                                                     const aiter_tensor_t& cos,
                                                     const aiter_tensor_t& sin,
                                                     const aiter_tensor_t& positions,
                                                     const aiter_tensor_t& slot_mapping,
                                                     float epsilon,
                                                     int32_t rope_dim,
                                                     int32_t kv_block_size = 16,
                                                     int32_t group_size = 32,
                                                     bool shuffle_scale = true,
                                                     bool do_rotate_act = false);

} // namespace aiter
