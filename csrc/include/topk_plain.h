#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_enum.h"
#include <torch/extension.h>

void topk_plain(torch::Tensor& values,
                torch::Tensor& topk_ids,
                torch::Tensor& topk_out,
                int topk,
                bool largest = true,
                torch::Tensor rowStarts = torch::Tensor(),
                torch::Tensor rowEnds = torch::Tensor(),
                int64_t stride0 = -1,
                int64_t stride1 = 1);
