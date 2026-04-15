// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

void top_k_per_row_prefill(const aiter_tensor_t& logits,
                           const aiter_tensor_t& rowStarts,
                           const aiter_tensor_t& rowEnds,
                           const aiter_tensor_t& indices,
                           int64_t numRows,
                           int64_t stride0,
                           int64_t stride1,
                           std::optional<aiter_tensor_t> values = std::nullopt);

void top_k_per_row_decode(const aiter_tensor_t& logits,
                          int64_t next_n,
                          const aiter_tensor_t& seqLens,
                          const aiter_tensor_t& indices,
                          int64_t numRows,
                          int64_t stride0,
                          int64_t stride1);
