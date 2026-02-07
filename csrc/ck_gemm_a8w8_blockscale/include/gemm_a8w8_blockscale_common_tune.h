// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>
#include <functional>
#include <unordered_map>

#include <torch/extension.h>

using BlockwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;

// For certain high priority shapes, we directly use the best kernel rather
// than use heuristics.
using BlockwiseKernelMap = std::unordered_map<int, BlockwiseKernel>;

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
    if(num <= 1)
        return 1;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}
