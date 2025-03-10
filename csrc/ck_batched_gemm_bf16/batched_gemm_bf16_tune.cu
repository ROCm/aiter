// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_bf16_common.cuh"
#include "batched_gemm_bf16_manifest.h"
#include "batched_gemm_bf16_lookup.h"
#include <string>

using BatchedKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &, std::optional<torch::Tensor>,
                  int)>;

// For certain high priority shapes, we directly use the best kernel rather
// than use heuristics.
using BatchedKernelMap = std::unordered_map<
    int,
    BatchedKernel>;

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

BatchedKernel batched_dispatch(int id)
{
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.

  // First check if this shape is available in the direct lookup.
  static const auto lookup = []
  {
      return BatchedKernelMap{GENERATE_LOOKUP_TABLE()};
  }();

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


torch::Tensor batched_gemm_bf16_tune(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    int kernelId,
    int splitK)
{
  TORCH_CHECK(XQ.dtype() == at::ScalarType::BFloat16 && XQ.dtype() == WQ.dtype(),
              "Weights and activations should both be bf16!");
  std::optional<torch::Tensor> bias = std::nullopt;

  int B = XQ.size(0);
  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = XQ.size(2);
  int KBatch = std::pow(2, splitK);

  if (Y.dtype() == at::ScalarType::BFloat16)
  {
    batched_dispatch(kernelId)(XQ, WQ, Y, bias, KBatch);
  }
  else
  {
    TORCH_CHECK(false, "Unsupported output dtype!");
  }
  return Y;
}
