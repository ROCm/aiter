// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& rowStarts,
                           const torch::Tensor& rowEnds,
                           torch::Tensor& indices,
                           std::optional<torch::Tensor> values,
                           int64_t numRows,
                           int64_t stride0,
                           int64_t stride1,
                           int64_t k                              = 2048,
                           std::optional<torch::Tensor> workspace = std::nullopt);

void top_k_per_row_decode(const torch::Tensor& logits,
                          int64_t next_n,
                          const torch::Tensor& seqLens,
                          torch::Tensor& indices,
                          int64_t numRows,
                          int64_t stride0,
                          int64_t stride1,
                          int64_t k                              = 2048,
                          std::optional<torch::Tensor> workspace = std::nullopt);

// Workspace-management queries exposed to Python (see get_topk_mb_workspace).
int64_t topk_mb_workspace_size(int64_t numRows, int64_t stride0, int64_t k, bool is_decode);
bool topk_use_mulblocks(int64_t numRows, int64_t stride0);
