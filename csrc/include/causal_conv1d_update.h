#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

namespace aiter {

void causal_conv1d_update(
    aiter_tensor_t& x,
    aiter_tensor_t& conv_state,
    aiter_tensor_t& weight,
    aiter_tensor_t& bias,
    aiter_tensor_t& out,
    bool use_silu,
    aiter_tensor_t& cache_seqlens,
    aiter_tensor_t& conv_state_indices,
    int pad_slot_id);

} // namespace aiter
