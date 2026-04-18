#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_enum.h"
#include "aiter_tensor.h"

void topk_plain(const aiter_tensor_t& values,
                const aiter_tensor_t& topk_ids,
                const aiter_tensor_t& topk_out,
                int topk,
                bool largest = true,
                const aiter_tensor_t* rowStarts = nullptr,
                const aiter_tensor_t* rowEnds = nullptr,
                int64_t stride0 = -1,
                int64_t stride1 = 1);
