// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

#include <optional>

void opus_moe_stage2_a8w4_decode_fwd(
    aiter_tensor_t& inter_states,
    aiter_tensor_t& w2,
    aiter_tensor_t& a2_scale,
    aiter_tensor_t& w2_scale,
    aiter_tensor_t& sorted_token_ids,
    std::optional<aiter_tensor_t> sorted_weights,
    aiter_tensor_t& sorted_expert_ids,
    aiter_tensor_t& num_valid_ids,
    aiter_tensor_t& out,
    int block_m,
    int kernel_id,
    int inter_dim_pad);

void opus_moe_stage2_reduce_token_slot_route_output_fwd(aiter_tensor_t& route_out,
                                                        aiter_tensor_t& out,
                                                        int topk,
                                                        int block_n);
