// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

void top_k_per_row_prefill(const aiter_tensor_t& logits,
                           const aiter_tensor_t& rowStarts,
                           const aiter_tensor_t& rowEnds,
                           aiter_tensor_t& indices,
                           std::optional<aiter_tensor_t> values,
                           int64_t numRows,
                           int64_t stride0,
                           int64_t stride1,
                           int64_t k                               = 2048,
                           std::optional<aiter_tensor_t> workspace = std::nullopt,
                           bool stable                             = false);

void top_k_per_row_decode(const aiter_tensor_t& logits,
                          int64_t next_n,
                          const aiter_tensor_t& seqLens,
                          aiter_tensor_t& indices,
                          int64_t numRows,
                          int64_t stride0,
                          int64_t stride1,
                          int64_t k                               = 2048,
                          std::optional<aiter_tensor_t> workspace = std::nullopt,
                          bool stable                             = false);

// Workspace-management queries exposed to Python (see get_topk_mb_workspace).
int64_t topk_mb_workspace_size(int64_t numRows, int64_t stride0, int64_t k, bool is_decode);
int64_t topk_ob_workspace_size(int64_t numRows, int64_t stride0, int64_t k, bool is_decode);
bool topk_use_mulblocks(int64_t numRows, int64_t stride0);
