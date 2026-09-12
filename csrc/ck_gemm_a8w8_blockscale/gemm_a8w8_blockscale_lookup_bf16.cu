// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// BF16 output dtype lookup table — separate TU so ninja compiles this and
// gemm_a8w8_blockscale_lookup_fp16.cu in parallel.

#include <string_view>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_a8w8_blockscale_common.cuh"
#include "gemm_a8w8_blockscale_lookup.h"
#include "gemm_a8w8_blockscale_manifest.h"

using BlockwiseKernel = torch::Tensor (*)(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, int);

using BlockwiseKernelMap = std::unordered_map<std::string_view, BlockwiseKernel>;

const BlockwiseKernelMap& get_blockscale_lookup_bf16()
{
    static const BlockwiseKernelMap map{GENERATE_LOOKUP_TABLE(FP32, BF16)};
    return map;
}
