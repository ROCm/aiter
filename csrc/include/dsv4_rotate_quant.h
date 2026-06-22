// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

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

// rope+hadamard. When `out_scale` is provided, additionally fp8-quantize the
// result: `out` is fp8 and `out_scale` receives per-(row, 1xGROUP) fp32 scales
// ([m, dim/group_size]), matching get_hip_quant(per_1x128). Without `out_scale`
// it is the bf16/fp16 in-place path (out shares dtype/stride with input).
void rope_rotate_activation(aiter_tensor_t& out,
                            const aiter_tensor_t& input,
                            const aiter_tensor_t& cos,
                            const aiter_tensor_t& sin,
                            const aiter_tensor_t& positions,
                            int32_t rope_dim,
                            std::optional<aiter_tensor_t> out_scale = std::nullopt,
                            int32_t group_size = 128);

} // namespace aiter
