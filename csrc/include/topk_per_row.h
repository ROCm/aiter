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
                          bool stable                             = false,
                          std::optional<aiter_tensor_t> values    = std::nullopt);

// Second fp32 prefill selector, from the topk-prefill-avo repo (generated into
// csrc/kernels/topk_per_row_sampled_kernels.cu; see the banner there before editing
// it). Same call shape as top_k_per_row_prefill so the two are interchangeable
// at the Python layer and directly comparable under one perftest, with two
// restrictions it reports rather than assumes: indices only (no `values`), and
// rows selected over their full `stride0` extent (rowStarts/rowEnds contents are
// not yet honoured). Ask topk_sampled_supports() before calling.
void top_k_per_row_prefill_sampled(const aiter_tensor_t& logits,
                               const aiter_tensor_t& rowStarts,
                               const aiter_tensor_t& rowEnds,
                               aiter_tensor_t& indices,
                               std::optional<aiter_tensor_t> values,
                               int64_t numRows,
                               int64_t stride0,
                               int64_t stride1,
                               int64_t k                               = 2048,
                               std::optional<aiter_tensor_t> workspace = std::nullopt);

int64_t topk_sampled_workspace_size(int64_t numRows, int64_t stride0, int64_t k);
bool topk_sampled_supports(int64_t numRows, int64_t stride0, int64_t k);

// Workspace-management queries exposed to Python (see get_topk_mb_workspace).
int64_t topk_mb_workspace_size(int64_t numRows, int64_t stride0, int64_t k, bool is_decode);
int64_t topk_ob_workspace_size(int64_t numRows, int64_t stride0, int64_t k, bool is_decode);
bool topk_use_mulblocks(int64_t numRows, int64_t stride0);
