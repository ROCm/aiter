#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"

#include <cstdint>

namespace aiter {

void chunk_gated_delta_rule_fwd_h_hip(
    aiter_tensor_t k,
    aiter_tensor_t w,
    aiter_tensor_t u,
    aiter_tensor_t g,
    aiter_tensor_t gk,
    aiter_tensor_t initial_state,
    aiter_tensor_t cu_seqlens,
    aiter_tensor_t chunk_offsets,
    aiter_tensor_t h,
    aiter_tensor_t v_new,
    aiter_tensor_t final_state,
    int64_t selected_bv,
    bool has_initial_state,
    bool output_final_state,
    bool save_new_value,
    bool use_exp2,
    bool g_head_major);

} // namespace aiter
