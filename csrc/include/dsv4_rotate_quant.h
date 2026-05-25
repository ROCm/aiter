// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

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

void rope_rotate_activation(aiter_tensor_t& out,
                            const aiter_tensor_t& input,
                            const aiter_tensor_t& cos,
                            const aiter_tensor_t& sin,
                            const aiter_tensor_t& positions,
                            int32_t rope_dim);

void rmsnorm_rope_rotate_activation_fp4quant_kvcache(aiter_tensor_t& kvcache,
                                                     aiter_tensor_t& scale,
                                                     const aiter_tensor_t& input,
                                                     const aiter_tensor_t& norm_weight,
                                                     const aiter_tensor_t& cos,
                                                     const aiter_tensor_t& sin,
                                                     const aiter_tensor_t& positions,
                                                     float epsilon,
                                                     int32_t rope_dim,
                                                     int32_t kv_block_size = 16,
                                                     int32_t group_size = 32,
                                                     bool shuffle_scale = true,
                                                     bool do_rotate_act = false);

} // namespace aiter
