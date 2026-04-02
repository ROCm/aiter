// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

void mha_prefill_splitk_stage1_opus(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& kv_indptr,
    const torch::Tensor& kv_page_indices,
    int32_t page_size,
    const torch::Tensor& work_indptr,
    const torch::Tensor& work_info_set,
    float softmax_scale,
    torch::Tensor& split_o,
    torch::Tensor& split_lse,
    std::optional<torch::Tensor> debug_qk_scores = std::nullopt);
