// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "deepgemm_common.cuh"
#include "deepgemm_lookup.h"
#include "deepgemm_manifest.h"
#include <cmath>
#include "py_itfs_common.h"

using RowwiseKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &, torch::Tensor &,
                  std::optional<torch::Tensor>, std::optional<torch::Tensor>)>;

// Define a custom hash function for std::tuple<int, int, int>
struct IntTupleHash
{
  size_t operator()(const std::tuple<int, int, int> &t) const
  {
    auto hash1 = std::hash<int>{}(std::get<0>(t));
    auto hash2 = std::hash<int>{}(std::get<1>(t));
    auto hash3 = std::hash<int>{}(std::get<2>(t));
    return hash1 ^ hash2 ^ hash3;
  }
};

// For certain high priority shapes, we directly use the best kernel rather
// than use heuristics.
using RowwiseKernelMap = std::unordered_map<
    std::tuple<int, int, int>,
    RowwiseKernel,
    IntTupleHash>;

template <typename ABDataType, typename AccDataType, typename CDataType>
RowwiseKernel rowwise_heuristic_dispatch(int M, int N, int K)
{
  // Apply shape heuristics to find a suitable kernel implementation.
  if (M < 128) 
  {
    return deepgemm_256x32x64x256_16x16x64_1x4<ABDataType, AccDataType, CDataType>;
  }
  else
  {
    return deepgemm_256x128x128x128_16x16x64_1x4<ABDataType, AccDataType, CDataType>;
  }
}

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename ABDataType, typename AccDataType, typename CDataType>
RowwiseKernel rowwise_dispatch(int M, int N, int K)
{
  // TODO: add tuner @lalala-sh
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.

  // static const auto lookup = [&]
  // {
  //   return RowwiseKernelMap{GENERATE_LOOKUP_TABLE(ABDataType, AccDataType, CDataType)};
  // }();

  // // First check if this shape(M,N,K) is available in the direct lookup.
  // auto it = lookup.find({M, N, K});
  // // If we found an optimal kernel, use it.
  // if (it != lookup.end())
  // {
  //   return it->second;
  // }

  // int padded_m = M;
  // if (M > 1 && M <= 16)
  // {
  //   padded_m = 16;
  // }
  // else if (M <= 16384)
  // {
  //   padded_m = nextPow2(M);
  // }
  // else if (M <= 20480)
  // {
  //   padded_m = 20480;
  // }
  // // Second check if this shape(padded_m,N,K) is available in the direct lookup.
  // it = lookup.find({padded_m, N, K});
  // // If we found an optimal kernel, use it.
  // if (it != lookup.end())
  // {
  //   return it->second;
  // }
  // Otherwise, use heuristics.
  return rowwise_heuristic_dispatch<ABDataType, AccDataType, CDataType>(M, N, K);
}

torch::Tensor deepgemm(
  torch::Tensor &XQ,
  torch::Tensor &WQ,
  torch::Tensor &Y,
  torch::Tensor &grouped_layout,
  std::optional<torch::Tensor> x_scale,
  std::optional<torch::Tensor> w_scale)
{
  TORCH_CHECK(XQ.dtype() == WQ.dtype(),
              "Weights and activations should both be int8/fp8!");
  if (x_scale != std::nullopt && w_scale != std::nullopt)
    TORCH_CHECK(x_scale.value().dtype() == w_scale.value().dtype(),
                "Scales should have the same dtype!");

  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);
  int KBatch = 1;



  if (XQ.dtype() == at::ScalarType::BFloat16 || XQ.dtype() == at::ScalarType::Half)
  {
    if (XQ.dtype() == at::ScalarType::Half)
    {
      rowwise_dispatch<fp16, float, fp16>(M, N, K)(XQ, WQ, Y, grouped_layout, x_scale, w_scale);
    }
    else
    {
      rowwise_dispatch<bf16, float, bf16>(M, N, K)(XQ, WQ, Y, grouped_layout, x_scale, w_scale);
    }
  }
  else if (XQ.dtype() == torch_fp8)
  {
    if (Y.dtype() == at::ScalarType::Half)
    {
      rowwise_dispatch<fp8, float, fp16>(M, N, K)(XQ, WQ, Y, grouped_layout, x_scale, w_scale);
    }
    else if (Y.dtype() == at::ScalarType::BFloat16)
    {
      rowwise_dispatch<fp8, float, bf16>(M, N, K)(XQ, WQ, Y, grouped_layout, x_scale, w_scale);
    }
  }
  else
  {
    TORCH_CHECK(false, "Unsupported scales/output dtype!");
  }
  return Y;
}