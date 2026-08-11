#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"

namespace aiter {

void gdr_decode_packed_bf16(aiter_tensor_t& mixed_qkv,
                            aiter_tensor_t& a,
                            aiter_tensor_t& b,
                            aiter_tensor_t& dt_bias,
                            aiter_tensor_t& A_log,
                            aiter_tensor_t& indices,
                            aiter_tensor_t& state,
                            aiter_tensor_t& out,
                            float scale);

} // namespace aiter
