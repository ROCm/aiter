// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>
#include <functional>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_a8w8_blockscale_cktile_common.cuh"
#include "gemm_a8w8_blockscale_cktile_lookup.h"
#include "gemm_a8w8_blockscale_cktile_manifest.h"

using BlockwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, bool)>;

// For certain high priority shapes, we directly use the best kernel rather
// than use heuristics.
using BlockwiseKernelMap = std::unordered_map<int, BlockwiseKernel>;

template <typename DDataType, typename EDataType = DDataType>
static BlockwiseKernel blockwise_dispatch_cktile(int id)
{
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.

    // First check if this shape is available in the direct lookup.
    static const auto lookup = [] {
        if constexpr(std::is_same_v<EDataType, TILE_FP16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, TILE_FP16)};
        }
        else if constexpr(std::is_same_v<EDataType, TILE_BF16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, TILE_BF16)};
        }
        else
        {
            static_assert(false, "blockwise_dispatch_cktile used with unsupported dtype!");
        }
    }();

    TORCH_CHECK(id < lookup.size(), "Kernel id " + std::to_string(id) + " is out of range!");
    auto it = lookup.find(id);
    // If we found an optimal kernel, use it.
    if(it != lookup.end())
    {
        return it->second;
    }
    // Otherwise, use heuristics.
    return lookup.find(0)->second;
}

torch::Tensor gemm_a8w8_blockscale_cktile_tune(torch::Tensor& XQ,
                                               torch::Tensor& WQ,
                                               torch::Tensor& x_scale,
                                               torch::Tensor& w_scale,
                                               torch::Tensor& Y,
                                               int kernelId,
                                               int splitK,
                                               bool preshuffleB)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");
    std::optional<torch::Tensor> bias = std::nullopt;

    int M      = XQ.size(0);
    int N      = WQ.size(0);
    int K      = XQ.size(1);
    int KBatch = std::pow(2, splitK);

    if(Y.dtype() == at::ScalarType::BFloat16)
    {
        blockwise_dispatch_cktile<TILE_FP32, TILE_BF16>(kernelId)(
            XQ, WQ, x_scale, w_scale, Y, preshuffleB);
    }
    else if(Y.dtype() == at::ScalarType::Half)
    {
        blockwise_dispatch_cktile<TILE_FP32, TILE_FP16>(kernelId)(
            XQ, WQ, x_scale, w_scale, Y, preshuffleB);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}

torch::Tensor gemm_a8w8_blockscale_bpreshuffle_cktile_tune(torch::Tensor& XQ,
                                                           torch::Tensor& WQ,
                                                           torch::Tensor& x_scale,
                                                           torch::Tensor& w_scale,
                                                           torch::Tensor& Y,
                                                           int kernelId,
                                                           int splitK,
                                                           bool preshuffleB)
{
    return gemm_a8w8_blockscale_cktile_tune(
        XQ, WQ, x_scale, w_scale, Y, kernelId, splitK, preshuffleB);
}
