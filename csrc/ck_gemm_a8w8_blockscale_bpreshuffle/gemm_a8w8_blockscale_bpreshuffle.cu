// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <string>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_a8w8_blockscale_bpreshuffle_common.cuh"
#include "gemm_a8w8_blockscale_bpreshuffle_lookup.h"
#include "gemm_a8w8_blockscale_bpreshuffle_manifest.h"

using BlockwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;

using BlockwiseKernelMap = std::unordered_map<std::string, BlockwiseKernel>;

// Python-driven name-keyed dispatch (see gemm_a8w8_blockscale.cu for the
// rationale).  Empty kernelName -> default heuristic; non-empty but unknown
// kernelName -> hard error.
template <typename DDataType, typename EDataType = DDataType>
BlockwiseKernel blockscale_bpreshuffle_dispatch(const std::string& kernelName)
{
    static const auto lookup = [] {
        if constexpr(std::is_same_v<EDataType, F16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, F16)};
        }
        else if constexpr(std::is_same_v<EDataType, B16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType, B16)};
        }
        else
        {
            static_assert(false, "blockscale_bpreshuffle_dispatch used with unsupported dtype!");
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
                    "gemm_a8w8_blockscale_bpreshuffle kernel '",
                    kernelName,
                    "' is not present in the compiled registry. The tuned CSV references a "
                    "kernel that was not built into aiter. Rebuild aiter (or remove this row "
                    "from aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv) and try "
                    "again.");
    }

    // Default heuristic kernel (used when Python had no tuned row).
    return a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<
        DDataType,
        EDataType>;
}

torch::Tensor gemm_a8w8_blockscale_bpreshuffle(torch::Tensor& XQ,
                                               torch::Tensor& WQ,
                                               torch::Tensor& x_scale,
                                               torch::Tensor& w_scale,
                                               torch::Tensor& Y,
                                               std::string kernelName)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
        blockscale_bpreshuffle_dispatch<F32, F16>(kernelName)(XQ, WQ, x_scale, w_scale, Y);
    }
    else if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
        blockscale_bpreshuffle_dispatch<F32, B16>(kernelName)(XQ, WQ, x_scale, w_scale, Y);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}
