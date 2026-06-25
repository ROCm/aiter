// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <string>
#include <string_view>
#include <unordered_map>

#include "gemm_a4w4_blockscale.h"

#include "gemm_a4w4_blockscale_common.cuh"
#include "gemm_a4w4_blockscale_lookup.h"
#include "gemm_a4w4_blockscale_manifest.h"

using BlockwiseKernel = aiter_tensor_t& (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    int, hipStream_t);

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
        if constexpr(std::is_same_v<CDataType, F16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(F16)};
        }
        else if constexpr(std::is_same_v<CDataType, B16>)
        {
            return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(B16)};
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
        AITER_CHECK(false,
                    "gemm_a4w4_blockscale kernel '",
                    kernelName,
                    "' is not present in the compiled registry. The tuned CSV references a "
                    "kernel that was not built into aiter. Rebuild aiter (or remove this row "
                    "from aiter/configs/a4w4_blockscale_tuned_gemm.csv) and try again.");
    }

    // Default heuristic kernel (used when Python had no tuned row).
    return a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<
        CDataType>;
}

namespace aiter {

aiter_tensor_t& gemm_a4w4_blockscale(aiter_tensor_t& XQ,
                                      aiter_tensor_t& WQ,
                                      aiter_tensor_t& x_scale,
                                      aiter_tensor_t& w_scale,
                                      aiter_tensor_t& Y,
                                      int splitK,
                                      hipStream_t stream,
                                      std::string kernelName)
{
    AITER_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    AITER_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    if(Y.dtype() == AITER_DTYPE_fp16)
    {
        blockscale_dispatch<F16>(kernelName)(XQ, WQ, x_scale, w_scale, Y, splitK, stream);
    }
    else if(Y.dtype() == AITER_DTYPE_bf16)
    {
        blockscale_dispatch<B16>(kernelName)(XQ, WQ, x_scale, w_scale, Y, splitK, stream);
    }
    else
    {
        AITER_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}

} // namespace aiter
