// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_common.cuh"
#include "gemm_a8w8_blockscale_common.h"
#include "gemm_a8w8_blockscale_lookup.h"
#include "gemm_a8w8_blockscale_manifest.h"

template <typename DDataType, typename EDataType = DDataType>
static BlockwiseKernel blockscale_dispatch(int M, int N, int K)
{
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.

    static const auto lookup = [] {
        if constexpr(std::is_same_v<EDataType, FP16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, FP16)};
        }
        else if constexpr(std::is_same_v<EDataType, BF16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, BF16)};
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

    // Default legacy kernel
    return a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<
        DDataType,
        EDataType>;
}

torch::Tensor gemm_a8w8_blockscale(torch::Tensor& XQ,
                                   torch::Tensor& WQ,
                                   torch::Tensor& x_scale,
                                   torch::Tensor& w_scale,
                                   torch::Tensor& Y)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
        blockscale_dispatch<FP32, FP16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    }
    else if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
        blockscale_dispatch<FP32, BF16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}
