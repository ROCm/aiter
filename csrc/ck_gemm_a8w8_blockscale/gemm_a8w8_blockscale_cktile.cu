// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>
#include <functional>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_common.h"

#include "gemm_a8w8_blockscale_cktile_common.cuh"
#include "gemm_a8w8_blockscale_cktile_lookup.h"
#include "gemm_a8w8_blockscale_cktile_manifest.h"

using BlockwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, bool)>;

// Define a custom hash function for std::tuple<int, int, int>
struct IntTupleHash
{
    size_t operator()(const std::tuple<int, int, int>& t) const
    {
        auto hash1 = std::hash<int>{}(std::get<0>(t));
        auto hash2 = std::hash<int>{}(std::get<1>(t));
        auto hash3 = std::hash<int>{}(std::get<2>(t));
        return hash1 ^ hash2 ^ hash3;
    }
};

using BlockwiseKernelMap =
    std::unordered_map<std::tuple<int, int, int>, BlockwiseKernel, IntTupleHash>;

template <typename DDataType, typename EDataType = DDataType>
static BlockwiseKernel blockscale_dispatch(int M, int N, int K)
{
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.

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
            static_assert(false, "blockscale_dispatch used with unsupported dtype!");
        }
    }();

    // First check if this shape(M,N,K) is available in the direct lookup.
    auto it = lookup.find({M, N, K});
    // If we found an optimal kernel, use it.
    if(it != lookup.end())
    {
        return it->second;
    }

    int padded_m = M;

    // Fine-grained search
    padded_m = getPaddedM(M, N, K, 0);

    // Second check if this shape(padded_m,N,K) is available in the direct lookup.
    it = lookup.find({padded_m, N, K});
    // If we found an optimal kernel, use it.
    if(it != lookup.end())
    {
        return it->second;
    }

    // Coarse-grained search
    padded_m = getPaddedM(M, N, K, 1);
    it       = lookup.find({padded_m, N, K});
    if(it != lookup.end())
    {
        return it->second;
    }

    // Default tile kernel
    return a8w8_blockscale_cktile_128x128x128_1x4x1_16x16x64_intrawave_0x1x0_1<DDataType,
                                                                               EDataType>;
}

torch::Tensor gemm_a8w8_blockscale_cktile(torch::Tensor& XQ,
                                          torch::Tensor& WQ,
                                          torch::Tensor& x_scale,
                                          torch::Tensor& w_scale,
                                          torch::Tensor& Y,
                                          bool preshuffleB)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
        blockscale_dispatch<TILE_FP32, TILE_FP16>(M, N, K)(
            XQ, WQ, x_scale, w_scale, Y, preshuffleB);
    }
    else if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
        blockscale_dispatch<TILE_FP32, TILE_BF16>(M, N, K)(
            XQ, WQ, x_scale, w_scale, Y, preshuffleB);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}

torch::Tensor gemm_a8w8_blockscale_bpreshuffle_cktile(torch::Tensor& XQ,
                                                      torch::Tensor& WQ,
                                                      torch::Tensor& x_scale,
                                                      torch::Tensor& w_scale,
                                                      torch::Tensor& Y,
                                                      bool preshuffleB)
{
    return gemm_a8w8_blockscale_cktile(XQ, WQ, x_scale, w_scale, Y, preshuffleB);
}
