// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

// Inputs as written by gfx1201_sage_prepare_hip. out: [B, S_pad, H, 128] bf16; rows >= seq_len are
// not written.
void gfx1201_sage_attention_fwd_hip(const aiter_tensor_t& q_int8,
                                    const aiter_tensor_t& q_scale,
                                    const aiter_tensor_t& k_int8,
                                    const aiter_tensor_t& k_scale,
                                    const aiter_tensor_t& v_fp8,
                                    const aiter_tensor_t& v_scale,
                                    aiter_tensor_t& out,
                                    int64_t seq_len);
