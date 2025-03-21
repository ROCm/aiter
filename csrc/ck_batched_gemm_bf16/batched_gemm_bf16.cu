// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_bf16_common.cuh"
#include "batched_gemm_bf16_manifest.h"
#include "batched_gemm_bf16_lookup.h"
#include <cmath>

using BatchedKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &, std::optional<torch::Tensor>,
                  int)>;

// Define a custom hash function for std::tuple<int, int, int, int>
struct IntTupleHash
{
  size_t operator()(const std::tuple<int, int, int, int> &t) const
  {
    auto hash1 = std::hash<int>{}(std::get<0>(t));
    auto hash2 = std::hash<int>{}(std::get<1>(t));
    auto hash3 = std::hash<int>{}(std::get<2>(t));
    auto hash4 = std::hash<int>{}(std::get<3>(t));
    return hash1 ^ hash2 ^ hash3 ^ hash4;
  }
};

// For certain high priority shapes, we directly use the best kernel rather
// than use heuristics.
using BatchedKernelMap = std::unordered_map<
    std::tuple<int, int, int, int>,
    BatchedKernel,
    IntTupleHash>;

BatchedKernel batched_heuristic_dispatch(int B, int M, int N, int K)
{
  // Apply shape heuristics to find a suitable kernel implementation.
  if (M < 64 && N < 2048 && K < 2048)
  {
    // Kernel that generally works well on small shapes.
    return bf16_batched_64x16x16x64_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2;
  }
  else if (M < 64 && K < 2048)
  {
    // Kernel that works well for small batch size and small K.
    return bf16_batched_128x16x32x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2;
  }
  else if (M < 64 && N < 2048)
  {
    // Kernel that works well for small batch size and small N.
    return bf16_batched_128x32x16x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2;
  }
  else if (M < 64 && N > 2048 && K > 2048)
  {
    // Kernel that works well for small M but larger N and K.
    return bf16_batched_64x16x16x128_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1;
  }
  else if (M < 64)
  {
    // Fallback to generic small batch kernel if we cant find a good match.
    return bf16_batched_64x16x16x64_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2;
    /* } else if (((M < 512 && K < 8192) || (N <= 2048 && K <= 8192) || (K <= 2048 && N <= 8192)) && K >= 1024) {
      // Kernel that is optimized for larger batch sizes but otherwise small
      // tensors.
      return bf16_batched_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5; */
  }
  else if (K < 1024)
  {
    // Special case for small K.
    return bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1;
  }
  else if (M < 1024)
  {
    // Kernel for generic medium batch sizes.
    return bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  else if (M >= 1024 && N >= 1024 && K >= 1024)
  {
    // Kernel for very large gemm
    // return bf16_batched_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
    return bf16_batched_256x256x128x32_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1;
  }
  else
  {
    // Fallback large kernel.
    return bf16_batched_256x224x256x32_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3;
  }
}

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

BatchedKernel batched_dispatch(int B, int M, int N, int K)
{
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.
  
  static const auto lookup = []
  {
      return BatchedKernelMap{GENERATE_LOOKUP_TABLE()};
  }();
  
  // First check if this shape(M,N,K) is available in the direct lookup.
  auto it = lookup.find({B, M, N, K});
  // If we found an optimal kernel, use it.
  if (it != lookup.end())
  {
    return it->second;
  }

  int padded_m = M;
  if (M > 1 && M <= 16)
  {
    padded_m = 16;
  }
  else if (M <= 16384)
  {
    padded_m = nextPow2(M);
  }
  else if (M <= 20480)
  {
    padded_m = 20480;
  }
  // Second check if this shape(padded_m,N,K) is available in the direct lookup.
  it = lookup.find({B, padded_m, N, K});
  // If we found an optimal kernel, use it.
  if (it != lookup.end())
  {
    return it->second;
  }
  // Otherwise, use heuristics.
  return batched_heuristic_dispatch(B, M, N, K);
}

torch::Tensor batched_gemm_bf16(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int splitK)
{
  TORCH_CHECK(XQ.dtype() == at::ScalarType::BFloat16 && XQ.dtype() == WQ.dtype(),
              "Weights and activations should both be bf16!");
  if (bias != std::nullopt)
    TORCH_CHECK(bias.value().dtype() == Y.dtype(),
                "Out amd bias should have the same dtype!");

  int B = XQ.size(0);
  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = XQ.size(2);
  int KBatch = 1;

  batched_dispatch(B, M, N, K)(XQ, WQ, Y, bias, KBatch);

  return Y;
}
