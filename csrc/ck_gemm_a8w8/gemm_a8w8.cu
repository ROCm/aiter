// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"
#include "gemm_a8w8_manifest.h"
#include "gemm_a8w8_lookup.h"
#include <cmath>

using RowwiseKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &, torch::Tensor &, 
                  torch::Tensor &, std::optional<torch::Tensor>,
                  int)>;

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

template <
  typename ADataType,
  typename BDataType,
  typename AccDataType,
  typename CShuffleDataType,
  typename ComputeDataType,
  typename DDataType,
  typename EDataType
>
RowwiseKernel rowwise_heuristic_dispatch(int M, int N, int K)
{
  // Apply shape heuristics to find a suitable kernel implementation.
  if (M < 64 && N < 2048 && K < 2048)
  {
    // Kernel that generally works well on small shapes.
    return a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<
      ADataType,
      BDataType,
      AccDataType,
      CShuffleDataType,
      ComputeDataType,
      DDataType,
      EDataType
    >;
  }
  else if (M < 64 && K < 2048)
  {
    // Kernel that works well for small batch size and small K.
    return a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
      ADataType,
      BDataType,
      AccDataType,
      CShuffleDataType,
      ComputeDataType,
      DDataType,
      EDataType
    >;
  }
  else if (M < 64 && N < 2048)
  {
    // Kernel that works well for small batch size and small N.
    return a8w8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2<
      ADataType,
      BDataType,
      AccDataType,
      CShuffleDataType,
      ComputeDataType,
      DDataType,
      EDataType
    >;
  }
  else if (M < 64 && N > 2048 && K > 2048)
  {
    // Kernel that works well for small M but larger N and K.
    return a8w8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x16x1x4_4x4x1_1x1_intrawave_v1<
      ADataType,
      BDataType,
      AccDataType,
      CShuffleDataType,
      ComputeDataType,
      DDataType,
      EDataType
    >;
  }
  else if (M < 64)
  {
    // Fallback to generic small batch kernel if we cant find a good match.
    return a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<
      ADataType,
      BDataType,
      AccDataType,
      CShuffleDataType,
      ComputeDataType,
      DDataType,
      EDataType
    >;
    /* } else if (((M < 512 && K < 8192) || (N <= 2048 && K <= 8192) || (K <= 2048 && N <= 8192)) && K >= 1024) {
      // Kernel that is optimized for larger batch sizes but otherwise small
      // tensors.
      return a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5<DDataType, EDataType>; */
  }
  else if (K < 1024)
  {
    // Special case for small K.
    return a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
      ADataType,
      BDataType,
      AccDataType,
      CShuffleDataType,
      ComputeDataType,
      DDataType,
      EDataType
    >;
  }
  else if (M < 1024)
  {
    // Kernel for generic medium batch sizes.
    return a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
      ADataType,
      BDataType,
      AccDataType,
      CShuffleDataType,
      ComputeDataType,
      DDataType,
      EDataType
    >;
  }
  else if (M >= 1024 && N >= 1024 && K >= 1024)
  {
    // Kernel for very large gemm
    // return a8w8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DDataType, EDataType>;
    return a8w8_rowwise_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<
      ADataType,
      BDataType,
      AccDataType,
      CShuffleDataType,
      ComputeDataType,
      DDataType,
      EDataType
    >;
  }
  else
  {
    // Fallback large kernel.
    return a8w8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<ADataType,
      BDataType,
      AccDataType,
      CShuffleDataType,
      ComputeDataType,
      DDataType,
      EDataType
    >;
  }
}

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
RowwiseKernel rowwise_dispatch(int M, int N, int K)
{
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.
  static const auto lookup = []
  {

    if constexpr (std::is_same_v<EDataType, F16>) {
        return RowwiseKernelMap{GENERATE_LOOKUP_TABLE(
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
        return RowwiseKernelMap{GENERATE_LOOKUP_TABLE(
          ADataType,
          BDataType,
          AccDataType,
          CShuffleDataType,
          ComputeDataType,
          DDataType,
          B16
          )
        };
    } 
    else {
        static_assert(false, "rowwise_dispatch used with unsupported dtype!");
    } 
  }();

  // First check if this shape(M,N,K) is available in the direct lookup.
  auto it = lookup.find({M, N, K});
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
  it = lookup.find({padded_m, N, K});
  // If we found an optimal kernel, use it.
  if (it != lookup.end())
  {
    return it->second;
  }
  // Otherwise, use heuristics.
  return rowwise_heuristic_dispatch<
    ADataType,
    BDataType,
    AccDataType,
    CShuffleDataType,
    ComputeDataType,
    DDataType,
    EDataType
  >(M, N, K);
}

torch::Tensor gemm_a8w8(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int splitK)
{
  const auto is_int8 = XQ.dtype() == at::ScalarType::Char && XQ.dtype() == WQ.dtype();
  const auto is_fp8 = XQ.dtype() == at::ScalarType::Float8_e4m3fnuz && XQ.dtype() == WQ.dtype();
  TORCH_CHECK(is_int8 || is_fp8, "Weights and activations should both be int8 or fp8!");
  TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");
  if (bias != std::nullopt)
    TORCH_CHECK(bias.value().dtype() == Y.dtype(),
                "Out amd bias should have the same dtype!");

  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);
  int KBatch = std::pow(2, splitK);
  // TODO: simplify
  if(is_int8){
    if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
      rowwise_dispatch<I8, I8, I32, I8, I32, F32, F16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
      rowwise_dispatch<I8, I8, I32, I8, I32, F32, B16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else if (Y.dtype() == at::ScalarType::Half)
    {
      rowwise_dispatch<I8, I8, I32, I8, I32, F16, F16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else if (Y.dtype() == at::ScalarType::BFloat16)
    {
      rowwise_dispatch<I8, I8, I32, I8, I32, B16, B16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else
    {
      TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
  } else {
        if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
      rowwise_dispatch<FP8, FP8, F32, FP8, F32, F32, F16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
      rowwise_dispatch<FP8, FP8, F32, FP8, F32, F32, B16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else if (Y.dtype() == at::ScalarType::Half)
    {
      rowwise_dispatch<FP8, FP8, F32, FP8, F32, F16, F16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else if (Y.dtype() == at::ScalarType::BFloat16)
    {
      rowwise_dispatch<FP8, FP8, F32, FP8, F32, B16, B16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else
    {
      TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
  }
  
  return Y;
}
