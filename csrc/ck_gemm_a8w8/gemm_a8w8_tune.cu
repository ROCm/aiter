// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"
#include "gemm_a8w8_manifest.h"
#include "gemm_a8w8_lookup.h"
#include <string>

using RowwiseKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &, torch::Tensor &, 
                  torch::Tensor &, std::optional<torch::Tensor>,
                  int)>;

// For certain high priority shapes, we directly use the best kernel rather
// than use heuristics.
using RowwiseKernelMap = std::unordered_map<
    int,
    RowwiseKernel>;

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <
  typename ADataType,
  typename BDataType,
  typename AccDataType,
  typename CShuffleDataType,
  typename ComputeDataType,
  typename DDataType,
  typename EDataType
>
RowwiseKernel rowwise_dispatch(int id){
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.

  // First check if this shape is available in the direct lookup.
  static const auto lookup = []
  {
    if constexpr (std::is_same_v<EDataType, F16>) {
      return RowwiseKernelMap{
        GENERATE_LOOKUP_TABLE(
          ADataType,
          BDataType,
          AccDataType,
          CShuffleDataType,
          ComputeDataType,
          DDataType,
          F16
        )
      };
    } else if constexpr (std::is_same_v<EDataType, B16>) {
      return RowwiseKernelMap{
        GENERATE_LOOKUP_TABLE(
          ADataType,
          BDataType,
          AccDataType,
          CShuffleDataType,
          ComputeDataType,
          DDataType,
          B16
        )
      };
    } else {
      static_assert(false, "rowwise_dispatch used with unsupported dtype!");
    } 
  }();

  TORCH_CHECK(id < lookup.size(), "Kernel id " + std::to_string(id)  +" is out of range!");
  auto it = lookup.find(id);
  // If we found an optimal kernel, use it.
  if (it != lookup.end())
  {
    return it->second;
  }
  // Otherwise, use heuristics.
  return lookup.find(0)->second;
}


torch::Tensor gemm_a8w8_int8(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int kernelId,
    int KBatch)
{
    std::optional<torch::Tensor> bias = std::nullopt;
    rowwise_dispatch<I8, I8, I32, I8, I32, B16, B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    return Y;
}

torch::Tensor gemm_a8w8_fp8(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int kernelId,
    int KBatch)
{
    std::optional<torch::Tensor> bias = std::nullopt;
    rowwise_dispatch<FP8, FP8, F32, FP8, F32, B16, B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    return Y;
}

torch::Tensor gemm_a8w8_tune(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int kernelId,
    int splitK
  )
{
  const auto is_int8 = XQ.dtype() == at::ScalarType::Char && XQ.dtype() == WQ.dtype();
  const auto is_fp8 = XQ.dtype() == at::ScalarType::Float8_e4m3fnuz && XQ.dtype() == WQ.dtype();
  TORCH_CHECK(is_int8 || is_fp8, "Weights and activations should both be either int8 or fp8!");
  TORCH_CHECK( x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");
  std::optional<torch::Tensor> bias = std::nullopt;

  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);
  int KBatch = std::pow(2, splitK);

  if(Y.dtype() != at::ScalarType::BFloat16)
    TORCH_CHECK(false, "Unsupported scales/output dtype!");

  if(is_fp8)
    return gemm_a8w8_fp8(XQ, WQ, x_scale, w_scale, Y, kernelId, KBatch);

  return  gemm_a8w8_int8(XQ, WQ, x_scale, w_scale, Y, kernelId, KBatch);
}