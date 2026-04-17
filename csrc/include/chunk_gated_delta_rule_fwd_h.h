#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

namespace aiter {

std::vector<torch::Tensor> chunk_gated_delta_rule_fwd_h_hip(
    torch::Tensor k,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor g,
    torch::Tensor initial_state,
    torch::Tensor cu_seqlens,
    torch::Tensor chunk_offsets,
    int64_t selected_bv,
    bool has_initial_state,
    bool output_final_state,
    bool save_new_value);

} // namespace aiter
