#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_enum.h"
#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

// `topk_out` is optional where `topk_plain_values_optional` in topk_plain.py
// says so: the fp32 largest-first radix path has a build that never writes the
// selected values, and a caller that only wants the indices should not pay for
// a buffer it discards. Passing nullopt outside that is an error, not silence.
void topk_plain(aiter_tensor_t& values,
                aiter_tensor_t& topk_ids,
                std::optional<aiter_tensor_t> topk_out,
                int topk,
                bool largest                            = true,
                std::optional<aiter_tensor_t> rowStarts = std::nullopt,
                std::optional<aiter_tensor_t> rowEnds   = std::nullopt,
                int64_t stride0                         = -1,
                int64_t stride1                         = 1,
                std::optional<aiter_tensor_t> workspace = std::nullopt);

// Workspace sizing for the fp32 radix path, exposed to Python so the scratch can
// be allocated + cached on the Python side and passed into topk_plain.
int64_t topk_plain_workspace_size(int64_t numRows, int64_t stride0, int64_t k);
