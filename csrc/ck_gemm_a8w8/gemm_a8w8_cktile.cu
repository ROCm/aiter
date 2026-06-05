// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <string>
#include <string_view>
#include <unordered_map>

#include <torch/extension.h>

#include "gemm_a8w8_cktile_common.cuh"
#include "gemm_a8w8_cktile_lookup.h"
#include "gemm_a8w8_cktile_manifest.h"
#include "gemm_dispatch_utils.h"
#include "py_itfs_common.h"


using RowwiseKernel = torch::Tensor (*)(torch::Tensor&,
                                        torch::Tensor&,
                                        torch::Tensor&,
                                        torch::Tensor&,
                                        torch::Tensor&,
                                        std::optional<torch::Tensor>,
                                        int);

// For certain high priority shapes, we directly use the best kernel rather
// than use heuristics.
using RowwiseKernelMap = GemmDispatchMap<RowwiseKernel>;

template <typename ABDataType, typename DDataType, typename EDataType>
RowwiseKernel rowwise_heuristic_dispatch(int M, int N, int K)
{
    // TODO: fill this:
    return a8w8_cktile_128x128x128_1x4x1_16x16x64_intrawave_0x1x0_1<ABDataType, DDataType, EDataType>;
}

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}


template <typename ABDataType, typename DDataType, typename EDataType>
RowwiseKernel rowwise_dispatch(int M, int N, int K)
{
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.

  static const auto lookup = []
  {
    return RowwiseKernelMap{GENERATE_LOOKUP_TABLE(ABDataType, DDataType, EDataType)};
  }();

  const int cu_num           = get_device_cu_num();
  const std::string_view gfx = get_device_gfx();

  // First check if this shape(M,N,K) is available in the direct lookup.
  auto it = lookup.find({gfx, cu_num, M, N, K});
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
  it = lookup.find({gfx, cu_num, padded_m, N, K});
  // If we found an optimal kernel, use it.
  if (it != lookup.end())
  {
    return it->second;
  }
  // Otherwise, use heuristics.
  return rowwise_heuristic_dispatch<ABDataType, DDataType, EDataType>(M, N, K);
}


torch::Tensor gemm_a8w8_cktile(torch::Tensor& XQ,
                               torch::Tensor& WQ,
                               torch::Tensor& x_scale,
                               torch::Tensor& w_scale,
                               torch::Tensor& Y,
                               std::optional<torch::Tensor> bias,
                               int splitK)
{
  TORCH_CHECK((XQ.dtype() == at::ScalarType::Char || XQ.dtype() == torch_fp8) &&
                  XQ.dtype() == WQ.dtype(),
              "Weights and activations should both be int8/fp8!");
  TORCH_CHECK(x_scale.dtype() == w_scale.dtype(),
              "Scales should have the same dtype!");
  if (bias != std::nullopt)
    TORCH_CHECK(bias.value().dtype() == Y.dtype(),
                "Out amd bias should have the same dtype!");

  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);
  int KBatch = std::pow(2, splitK);

  // if (XQ.dtype() == at::ScalarType::Char)
  // {
  //   if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
  //   {
  //     rowwise_dispatch<TILE_I8, TILE_FP32, TILE_FP16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
  //   }
  //   else if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
  //   {
  //     rowwise_dispatch<TILE_I8, TILE_FP32, TILE_BF16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
  //   }
  //   else if (Y.dtype() == at::ScalarType::Half)
  //   {
  //     rowwise_dispatch<TILE_I8, TILE_FP16, TILE_FP16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
  //   }
  //   else if (Y.dtype() == at::ScalarType::BFloat16)
  //   {
  //     rowwise_dispatch<TILE_I8, TILE_BF16, TILE_BF16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
  //   }
  //   else
  //   {
  //     TORCH_CHECK(false, "Unsupported scales/output dtype!");
  //   }
  // }
  // else
  {
    if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::Half)
    {
      rowwise_dispatch<TILE_FP8, TILE_FP32, TILE_FP16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else if (x_scale.dtype() == at::ScalarType::Float && Y.dtype() == at::ScalarType::BFloat16)
    {
      rowwise_dispatch<TILE_FP8, TILE_FP32, TILE_BF16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else if (Y.dtype() == at::ScalarType::Half)
    {
      rowwise_dispatch<TILE_FP8, TILE_FP16, TILE_FP16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else if (Y.dtype() == at::ScalarType::BFloat16)
    {
      rowwise_dispatch<TILE_FP8, TILE_BF16, TILE_BF16>(M, N, K)(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }
    else
    {
      TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
  }
  return Y;
}