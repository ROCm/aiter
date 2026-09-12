#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"

#include <cstdint>

namespace aiter {

// No C++ torch dependency: tensors are passed as POD ``aiter_tensor_t`` (see
// ``chunk_gated_delta_rule_fwd_h``). Outputs ``q``/``k``/``v`` are pre-allocated
// by the Python caller and written in-place, so the entry point returns void.
void causal_conv1d_fwd_split_qkv_hip(
    aiter_tensor_t x,
    aiter_tensor_t weight,
    aiter_tensor_t bias,
    aiter_tensor_t conv_states,
    aiter_tensor_t cache_indices,
    aiter_tensor_t has_initial_state,
    aiter_tensor_t query_start_loc,
    aiter_tensor_t batch_ptr,
    aiter_tensor_t token_chunk_offset_ptr,
    aiter_tensor_t q,
    aiter_tensor_t k,
    aiter_tensor_t v,
    int64_t k_dim,
    int64_t v_dim,
    int64_t n_programs,
    int64_t block_m,
    bool has_bias,
    bool silu,
    int64_t pad_slot_id);

} // namespace aiter
