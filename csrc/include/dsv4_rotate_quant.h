// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

void rotate_activation_fp4quant_inplace(aiter_tensor_t& out,
                                        const aiter_tensor_t& input,
                                        int32_t group_size = 32);


void rotate_activation(aiter_tensor_t& out,
                       const aiter_tensor_t& input);

void rope_rotate_activation_fp4quant_inplace(aiter_tensor_t& out,
                                            const aiter_tensor_t& input,
                                            const aiter_tensor_t& cos,
                                            const aiter_tensor_t& sin,
                                            const aiter_tensor_t& positions,
                                            int32_t rope_dim,
                                            int32_t group_size = 32);

void rope_rotate_activation(aiter_tensor_t& out,
                            const aiter_tensor_t& input,
                            const aiter_tensor_t& cos,
                            const aiter_tensor_t& sin,
                            const aiter_tensor_t& positions,
                            int32_t rope_dim);

} // namespace aiter
