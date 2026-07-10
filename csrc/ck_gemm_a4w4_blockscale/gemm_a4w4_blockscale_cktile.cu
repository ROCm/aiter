// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <string>
#include <string_view>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_a4w4_blockscale_cktile_common.cuh"
#include "gemm_a4w4_blockscale_cktile_lookup.h"
#include "gemm_a4w4_blockscale_cktile_manifest.h"

using BlockwiseKernel = torch::Tensor (*)(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, int);

// Name-keyed dispatch table; see gemm_a8w8_blockscale.cu for the rationale
// behind std::string_view keys + raw fn-ptr values (constant-init into
// .data.rel.ro, matching PR #3255's GemmDispatchMap style).
//
// Note: the a4w4 tuned CSV mixes CK and ASM kernels; the ASM rows are
// filtered out at codegen (see csrc/ck_gemm_a4w4_blockscale/gen_instances.py)
// and dispatched separately by aiter/ops/gemm_op_a4w4.py based on the
// kernelName prefix (_ZN... = mangled ASM symbol).  This map only contains
// CK kernels.
using BlockwiseKernelMap = std::unordered_map<std::string_view, BlockwiseKernel>;

// Python-driven name-keyed dispatch (see gemm_a8w8_blockscale.cu for the
// rationale).  Empty kernelName -> default heuristic; non-empty but unknown
// kernelName -> hard error.
template <typename CDataType>
BlockwiseKernel blockscale_dispatch(const std::string& kernelName)
{
    static const auto lookup = [] {
        if constexpr(std::is_same_v<CDataType, TILE_FP16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(TILE_FP16)};
        }
        else if constexpr(std::is_same_v<CDataType, TILE_BF16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(TILE_BF16)};
        }
        else
        {
            static_assert(false, "blockscale_dispatch used with unsupported dtype!");
        }
    }();

    if(!kernelName.empty())
    {
        auto it = lookup.find(std::string_view{kernelName});
        if(it != lookup.end())
        {
            return it->second;
        }
        TORCH_CHECK(false,
                    "gemm_a4w4_blockscale kernel '",
                    kernelName,
                    "' is not present in the compiled registry. The tuned CSV references a "
                    "kernel that was not built into aiter. Rebuild aiter (or remove this row "
                    "from aiter/configs/a4w4_blockscale_tuned_gemm.csv) and try again.");
    }

    // Default heuristic kernel (used when Python had no tuned row).
    return a4w4_blockscale_cktile_128x512x256_1x4x1_16x16x128_default_0x0x0_1<
        CDataType>;
}

torch::Tensor gemm_a4w4_blockscale_cktile(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y,
    int splitK,
    std::string kernelName)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    if(Y.dtype() == at::ScalarType::Half)
    {
        blockscale_dispatch<TILE_FP16>(kernelName)(XQ, WQ, x_scale, w_scale, Y, splitK);
    }
    else if(Y.dtype() == at::ScalarType::BFloat16)
    {
        blockscale_dispatch<TILE_BF16>(kernelName)(XQ, WQ, x_scale, w_scale, Y, splitK);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}
