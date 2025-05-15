// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_wpreshuffle_common.cuh"
#include "gemm_a8w8_blockscale_wpreshuffle_manifest.h"
#include "gemm_a8w8_blockscale_wpreshuffle_lookup.h"
#include <string>
#include "py_itfs_common.h"

using BlockwiseKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &, torch::Tensor &, 
                  torch::Tensor &)>;

// For certain high priority shapes, we directly use the best kernel rather
// than use heuristics.
using BlockwiseKernelMap = std::unordered_map<
    int,
    BlockwiseKernel>;

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename DDataType, typename EDataType = DDataType>
BlockwiseKernel blockwise_dispatch(int id)
{
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.

  // First check if this shape is available in the direct lookup.
  static const auto lookup = []
  {
    if constexpr (std::is_same_v<EDataType, F16>) {
        return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType,F16)};
    } else if constexpr (std::is_same_v<EDataType, B16>) {
        return BlockwiseKernelMap{GENERATE_LOOKUP_TABLE(DDataType,B16)};
    } else {
        static_assert(false, "blockwise_dispatch used with unsupported dtype!");
    } }();

  TORCH_CHECK(id < lookup.size(),
              "Kernel id " + std::to_string(id)  +" is out of range!");
  auto it = lookup.find(id);
  // If we found an optimal kernel, use it.
  if (it != lookup.end())
  {
    return it->second;
  }
  // Otherwise, use heuristics.
  return lookup.find(0)->second;
}



torch::Tensor gemm_a8w8_blockscale_wpreshuffle_tune(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int kernelId,
    int splitK)
{
  TORCH_CHECK(XQ.dtype() == torch_fp8 && XQ.dtype() == WQ.dtype(),
              "Weights and activations should both be fp8!");
  TORCH_CHECK( x_scale.dtype() == w_scale.dtype(),
              "Scales should have the same dtype!");
  std::optional<torch::Tensor> bias = std::nullopt;

  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);
  int KBatch = std::pow(2, splitK);

  if (Y.dtype() == at::ScalarType::BFloat16)
  {
    blockwise_dispatch<F32, B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y);
  }
  else
  {
    TORCH_CHECK(false, "Unsupported scales/output dtype!");
  }
  return Y;
}
