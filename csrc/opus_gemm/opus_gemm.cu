// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "opus_gemm_common.cuh"
#include "opus_gemm_lookup.h"
#include "opus_gemm_manifest.h"
#include "opus_gemm_a16w16_tune_lookup.h"
#include "py_itfs_common.h"
#include <cmath>
#include <string>

// Scale kernel signature (a8w8_scale)
using OpusScaleKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &,
                  std::optional<torch::Tensor>, std::optional<torch::Tensor>)>;

// Noscale kernel signature (a8w8, a16w16)
using OpusNoscaleKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &)>;

// a8w8_scale dispatch
template <typename CDataType>
OpusScaleKernel opus_dispatch_scale(int M, int N, int K)
{
  return opus_gemm_512x256x256x128_4x2_16x16x128_1x128x128<CDataType>;
}

// a8w8 noscale dispatch
template <typename CDataType>
OpusNoscaleKernel opus_dispatch_a8w8(int M, int N, int K)
{
  return opus_gemm_512x256x256x128_2x4_16x16x128_0x0x0<CDataType>;
}

// a16w16 dispatch
template <typename CDataType>
OpusNoscaleKernel opus_dispatch_a16w16(int M, int N, int K)
{
  return opus_gemm_512x256x256x64_2x4_16x16x32_0x0x0<CDataType>;
}

torch::Tensor opus_gemm(
  torch::Tensor &XQ,
  torch::Tensor &WQ,
  torch::Tensor &Y,
  std::optional<torch::Tensor> group_layout,
  std::optional<torch::Tensor> x_scale,
  std::optional<torch::Tensor> w_scale)
{
  TORCH_CHECK(XQ.dim() == 3, "XQ must be 3D [batch, M, K]");
  TORCH_CHECK(WQ.dim() == 3, "WQ must be 3D [batch, N, K]");
  TORCH_CHECK(Y.dim() == 3, "Y must be 3D [batch, M, N]");

  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = XQ.size(2);

  bool has_scale = x_scale.has_value() && w_scale.has_value();

  if (XQ.dtype() == torch_fp8)
  {
    if (has_scale)
    {
      TORCH_CHECK(Y.dtype() == at::ScalarType::Float,
                  "opus_gemm a8w8_scale only supports fp32 output");
      opus_dispatch_scale<fp32_t>(M, N, K)(XQ, WQ, Y, x_scale, w_scale);
    }
    else
    {
      TORCH_CHECK(Y.dtype() == at::ScalarType::Float,
                  "opus_gemm a8w8 no-scale only supports fp32 output");
      opus_dispatch_a8w8<fp32_t>(M, N, K)(XQ, WQ, Y);
    }
  }
  else if (XQ.dtype() == at::ScalarType::BFloat16)
  {
    if (Y.dtype() == at::ScalarType::BFloat16)
    {
      opus_dispatch_a16w16<bf16_t>(M, N, K)(XQ, WQ, Y);
    }
    else if (Y.dtype() == at::ScalarType::Float)
    {
      opus_dispatch_a16w16<fp32_t>(M, N, K)(XQ, WQ, Y);
    }
    else
    {
      TORCH_CHECK(false, "opus_gemm a16w16: unsupported output dtype, expected bf16 or fp32");
    }
  }
  else
  {
    TORCH_CHECK(false, "opus_gemm: unsupported input dtype, expected fp8 or bf16");
  }
  return Y;
}

// ── a16w16 tune dispatch (id-based) ──

using OpusA16W16TuneKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &)>;

using OpusA16W16TuneMap = std::unordered_map<
    int,
    OpusA16W16TuneKernel>;

template <typename CDataType>
OpusA16W16TuneKernel opus_a16w16_tune_dispatch(int id)
{
  static const auto lookup = []
  {
    return OpusA16W16TuneMap{GENERATE_A16W16_TUNE_LOOKUP(CDataType)};
  }();

  auto it = lookup.find(id);
  TORCH_CHECK(it != lookup.end(),
              "Kernel id " + std::to_string(id) + " not found in a16w16 tune lookup table!");
  return it->second;
}

torch::Tensor opus_gemm_a16w16_tune(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    int kernelId,
    int splitK)
{
  TORCH_CHECK(XQ.dim() == 3, "XQ must be 3D [batch, M, K]");
  TORCH_CHECK(WQ.dim() == 3, "WQ must be 3D [batch, N, K]");
  TORCH_CHECK(Y.dim() == 3, "Y must be 3D [batch, M, N]");
  TORCH_CHECK(XQ.dtype() == WQ.dtype(),
              "XQ and WQ should have the same dtype!");

  if (XQ.dtype() == at::ScalarType::BFloat16)
  {
    if (Y.dtype() == at::ScalarType::BFloat16)
    {
      opus_a16w16_tune_dispatch<bf16_t>(kernelId)(XQ, WQ, Y);
    }
    else if (Y.dtype() == at::ScalarType::Float)
    {
      opus_a16w16_tune_dispatch<fp32_t>(kernelId)(XQ, WQ, Y);
    }
    else
    {
      TORCH_CHECK(false,
                  "opus_gemm_a16w16_tune: unsupported output dtype, expected bf16 or fp32");
    }
  }
  else
  {
    TORCH_CHECK(false,
                "opus_gemm_a16w16_tune: unsupported input dtype " +
                    std::string(c10::toString(XQ.dtype())) +
                    ", expected bf16");
  }
  return Y;
}
