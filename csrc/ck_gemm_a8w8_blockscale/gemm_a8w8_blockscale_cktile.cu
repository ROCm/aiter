// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <string>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_a8w8_blockscale_cktile_common.cuh"
#include "gemm_a8w8_blockscale_cktile_lookup.h"
#include "gemm_a8w8_blockscale_cktile_manifest.h"

using BlockwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, bool, int)>;

using BlockwiseKernelMap = std::unordered_map<std::string, BlockwiseKernel>;

// Python-driven name-keyed dispatch (see gemm_a8w8_blockscale.cu for the
// rationale).  Empty kernelName -> default heuristic; non-empty but unknown
// kernelName -> hard error.
template <typename DDataType, typename EDataType = DDataType>
static BlockwiseKernel blockscale_dispatch(const std::string& kernelName)
{
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

    if(!kernelName.empty())
    {
        auto it = lookup.find(kernelName);
        if(it != lookup.end())
        {
            return it->second;
        }
        TORCH_CHECK(false,
                    "gemm_a8w8_blockscale_cktile kernel '",
                    kernelName,
                    "' is not present in the compiled registry. The tuned CSV references a "
                    "kernel that was not built into aiter. Rebuild aiter (or remove this row "
                    "from aiter/configs/a8w8_blockscale_tuned_gemm.csv) and try again.");
    }

    // Default tile kernel (used when Python had no tuned row).
    return a8w8_blockscale_cktile_128x128x128_1x4x1_16x16x64_intrawave_0x1x0_1<DDataType, EDataType>;
}

torch::Tensor gemm_a8w8_blockscale_cktile(torch::Tensor& XQ,
                                          torch::Tensor& WQ,
                                          torch::Tensor& x_scale,
                                          torch::Tensor& w_scale,
                                          torch::Tensor& Y,
                                          bool preshuffleB,
                                          int splitK,
                                          std::string kernelName)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    TORCH_CHECK(splitK >= 0 && splitK <= 30,
                "splitK must be in the range [0, 30], got ",
                splitK);

    int KBatch = 1 << splitK;

    if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
        blockscale_dispatch<TILE_FP32, TILE_FP16>(kernelName)(
            XQ, WQ, x_scale, w_scale, Y, preshuffleB, KBatch);
    }
    else if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
        blockscale_dispatch<TILE_FP32, TILE_BF16>(kernelName)(
            XQ, WQ, x_scale, w_scale, Y, preshuffleB, KBatch);
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
                                                      bool preshuffleB,
                                                      std::string kernelName)
{
    return gemm_a8w8_blockscale_cktile(
        XQ, WQ, x_scale, w_scale, Y, preshuffleB, 0, std::move(kernelName));
}
