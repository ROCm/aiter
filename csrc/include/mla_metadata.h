// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

// Returns
//   [0] cumulated num_kv_splits: (batch_size + 1), dtype torch.int32.
//   [1] max_num_splits: (1), dtype torch.int32.
std::vector<torch::Tensor> get_mla_metadata_v0(const torch::Tensor& p_seqlens_k,
                                               const int32_t num_heads_per_head_k,
                                               const int32_t num_heads_k);
