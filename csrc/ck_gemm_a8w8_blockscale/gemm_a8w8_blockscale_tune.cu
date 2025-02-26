// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_common.cuh"
#include "gemm_a8w8_blockscale_manifest.h"
#include "gemm_a8w8_blockscale_lookup.h"
#include <string>

using BlockscaleKernel = std::function<
      torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &, torch::Tensor &, 
                  torch::Tensor &)>; 

// For certain high priority shapes, we directly use the best kernel rather
// than use heuristics.
using BlockscaleKernelMap = std::unordered_map<
    int, 
    BlockscaleKernel>; 

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename DDataType, typename EDataType = DDataType>
BlockscaleKernel blockscale_dispatch(int M, int N, int K)
{
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.
    
    static const auto lookup = []
    {
      if constexpr (std::is_same_v<EDataType, F16>) {
          return BlockscaleKernelMap{GENERATE_LOOKUP_TABLE(DDataType,F16)};
      } else if constexpr (std::is_same_v<EDataType, B16>) {
          return BlockscaleKernelMap{GENERATE_LOOKUP_TABLE(DDataType,B16)};
      } else {
          static_assert(false, "blockscale_dispatch used with unsupported dtype!");
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

torch::Tensor gemm_a8w8_blockscale_tune(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == Y.dtype() && w_scale.dtype() == Y.dtype(),
                  "Scales and output should have the same dtype!");

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

  if (Y.dtype() == at::ScalarType::BFloat16)
  {
    blcokscale_dispatch<B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y);
  }
  else
  {
    TORCH_CHECK(false, "Unsupported scales/output dtype!");
  }
  return Y;   
}