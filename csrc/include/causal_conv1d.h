#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

namespace aiter {

void causal_conv1d_update(
    const aiter_tensor_t& x,
    const aiter_tensor_t& conv_state,
    const aiter_tensor_t& weight,
    const aiter_tensor_t* bias,
    const aiter_tensor_t& out,
    bool use_silu,
    const aiter_tensor_t* cache_seqlens,
    const aiter_tensor_t* conv_state_indices,
    int pad_slot_id);

} // namespace aiter
