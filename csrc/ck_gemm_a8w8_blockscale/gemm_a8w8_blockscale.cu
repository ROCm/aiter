// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <string>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_a8w8_blockscale_common.cuh"
#include "gemm_a8w8_blockscale_lookup.h"
#include "gemm_a8w8_blockscale_manifest.h"

using BlockwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, int)>;

using BlockwiseKernelMap = std::unordered_map<std::string, BlockwiseKernel>;

// Python-driven name-keyed dispatch.  The Python frontend
// (aiter/ops/gemm_op_a8w8.py) reads aiter/configs/a8w8_blockscale_tuned_gemm.csv
// (cached via lru_cache) and passes the resolved `kernelName` here; we only
// look it up.  This makes the CSV the single source of truth — editing it no
// longer requires rebuilding a tuple-keyed C++ table.
//
// Empty kernelName  : Python had no tuned row (or AITER_BYPASS_TUNE_CONFIG=1)
//                     -> use the heuristic default kernel.
// Non-empty kernelName not in registry: hard error — the CSV references a
//                     kernel that was not compiled into this .so, almost
//                     certainly because the user updated the CSV without
//                     rebuilding aiter.  Surface, don't hide.
template <typename DDataType, typename EDataType = DDataType>
static BlockwiseKernel blockscale_dispatch(const std::string& kernelName)
{
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

    if(!kernelName.empty())
    {
        auto it = lookup.find(kernelName);
        if(it != lookup.end())
        {
            return it->second;
        }
        TORCH_CHECK(false,
                    "gemm_a8w8_blockscale kernel '",
                    kernelName,
                    "' is not present in the compiled registry. The tuned CSV references a "
                    "kernel that was not built into aiter. Rebuild aiter (or remove this row "
                    "from aiter/configs/a8w8_blockscale_tuned_gemm.csv) and try again.");
    }

    // Default legacy kernel (used when Python had no tuned row).
    return a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<
        DDataType,
        EDataType>;
}

torch::Tensor gemm_a8w8_blockscale(torch::Tensor& XQ,
                                   torch::Tensor& WQ,
                                   torch::Tensor& x_scale,
                                   torch::Tensor& w_scale,
                                   torch::Tensor& Y,
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
        blockscale_dispatch<FP32, FP16>(kernelName)(XQ, WQ, x_scale, w_scale, Y, KBatch);
    }
    else if(x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
        blockscale_dispatch<FP32, BF16>(kernelName)(XQ, WQ, x_scale, w_scale, Y, KBatch);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}
