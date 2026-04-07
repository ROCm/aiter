// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a4w4_blockscale_common.cuh"
#include "gemm_a4w4_blockscale_manifest.h"
#include "gemm_a4w4_blockscale_lookup.h"
#include "gemm_common.h"
#include "gemm_dispatch_utils.h"
#include <cmath>

using BlockwiseKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &, torch::Tensor &,
                  torch::Tensor &, int)>;

using BlockwiseKernelMap = GemmDispatchMap<BlockwiseKernel>;

template <typename CDataType>
BlockwiseKernel blockscale_dispatch(int M, int N, int K)
{
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.

    static const auto lookup = []
    {
      if constexpr (std::is_same_v<CDataType, F16>) {
          return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(F16)};
      } else if constexpr (std::is_same_v<CDataType, B16>) {
          return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(B16)};
      } else {
          static_assert(false, "blockscale_dispatch used with unsupported dtype!");
      } }();

    const int cu_num = get_device_cu_num();

    // First check if this shape(M,N,K) is available in the direct lookup.
    auto it = lookup.find({cu_num, M, N, K});
    // If we found an optimal kernel, use it.
    if (it != lookup.end())
    {
      return it->second;
    }

    int padded_m = M;
    // Fine-grained search
    padded_m = getPaddedM(M, N, K, 0);

    // Second check if this shape(padded_m,N,K) is available in the direct lookup.
    it = lookup.find({cu_num, padded_m, N, K});
    // If we found an optimal kernel, use it.
    if (it != lookup.end())
    {
      return it->second;
    }
    // Coarse-grained search
    padded_m = getPaddedM(M, N, K, 1);
    it = lookup.find({cu_num, padded_m, N, K});
    if (it != lookup.end())
    {
      return it->second;
    }
    // Otherwise, use heuristics.
    return a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CDataType>;
}

torch::Tensor gemm_a4w4_blockscale(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y,
    int splitK)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(),
                "Scales should have the same dtype!");

  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1) * 2; // always fp4_x2

  if (Y.dtype() == at::ScalarType::Half)
  {
    blockscale_dispatch<F16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, splitK);
  }
  else if (Y.dtype() == at::ScalarType::BFloat16)
  {
    blockscale_dispatch<B16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, splitK);
  }
  else
  {
    TORCH_CHECK(false, "Unsupported scales/output dtype!");
  }
  return Y;
}
