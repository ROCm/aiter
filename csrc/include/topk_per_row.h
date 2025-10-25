// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void topk_per_row(const torch::Tensor& logits,
                  const torch::Tensor& rowStarts,
                  const torch::Tensor& rowEnds,
                  torch::Tensor& indices,
                  int64_t numRows,
                  int64_t stride0,
                  int64_t stride1);

void topk_per_row_decode(const torch::Tensor& logits,
                         int64_t next_n,
                         const torch::Tensor& seqLens,
                         torch::Tensor& indices,
                         int64_t numRows,
                         int64_t stride0,
                         int64_t stride1);
