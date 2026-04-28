// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "opus_gemm_common.cuh"
#include "opus_gemm_heuristic_dispatch.cuh"  // OpusA16W16NoscaleKernel + opus_a16w16_heuristic_dispatch<>
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

// Noscale kernel signature (a8w8)
using OpusNoscaleKernel = std::function<
    torch::Tensor(torch::Tensor &, torch::Tensor &,
                  torch::Tensor &)>;

// a16w16-family launcher signature is provided by
// opus_gemm_heuristic_dispatch.cuh as OpusA16W16NoscaleKernel.

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

// ── a16w16 runtime dispatch (two-level: tuned lookup → heuristic) ──
//
// The heuristic fall-through (opus_a16w16_heuristic_dispatch<CDataType>)
// is defined in opus_gemm_heuristic_dispatch.cuh; the rest of this
// section wires it to the JIT-baked (M,N,K)->kernel lookup map.

// (M, N, K) -> kernel<CDataType>. Populated from opus-private tuned CSV
// at JIT time by gen_instances.py --tune_file. Mirrors the IntTupleHash
// + unordered_map layout used by csrc/ck_gemm_a8w8/gemm_a8w8.cu.
struct IntTupleHash
{
  size_t operator()(const std::tuple<int, int, int> &t) const
  {
    auto h1 = std::hash<int>{}(std::get<0>(t));
    auto h2 = std::hash<int>{}(std::get<1>(t));
    auto h3 = std::hash<int>{}(std::get<2>(t));
    return h1 ^ h2 ^ h3;
  }
};

using OpusA16W16RuntimeMap = std::unordered_map<
    std::tuple<int, int, int>,
    OpusA16W16NoscaleKernel,
    IntTupleHash>;

template <typename CDataType>
OpusA16W16NoscaleKernel opus_dispatch_a16w16(int M, int N, int K, int batch);

template <>
OpusA16W16NoscaleKernel opus_dispatch_a16w16<bf16_t>(int M, int N, int K, int batch)
{
  static const auto lookup = []
  {
    return OpusA16W16RuntimeMap{GENERATE_OPUS_LOOKUP_TABLE_BF16(bf16_t)};
  }();
  auto it = lookup.find({M, N, K});
  if (it != lookup.end())
  {
    return it->second;
  }
  return opus_a16w16_heuristic_dispatch<bf16_t>(M, N, K, batch);
}

template <>
OpusA16W16NoscaleKernel opus_dispatch_a16w16<fp32_t>(int M, int N, int K, int batch)
{
  static const auto lookup = []
  {
    return OpusA16W16RuntimeMap{GENERATE_OPUS_LOOKUP_TABLE_FP32(fp32_t)};
  }();
  auto it = lookup.find({M, N, K});
  if (it != lookup.end())
  {
    return it->second;
  }
  return opus_a16w16_heuristic_dispatch<fp32_t>(M, N, K, batch);
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
    // Two-level dispatch: tuned CSV lookup (baked in at JIT time) first,
    // heuristic if-else tree on miss. splitK is passed 0; splitk kids
    // auto-clamp to pfk anyway so no extra information is needed at this
    // entry point. Kernels that ignore splitK (a16w16 / flatmm) just
    // drop it.
    int batch = XQ.size(0);
    if (Y.dtype() == at::ScalarType::BFloat16)
    {
      opus_dispatch_a16w16<bf16_t>(M, N, K, batch)(XQ, WQ, Y, 0);
    }
    else if (Y.dtype() == at::ScalarType::Float)
    {
      opus_dispatch_a16w16<fp32_t>(M, N, K, batch)(XQ, WQ, Y, 0);
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
//
// Launcher signature is 4-arg (XQ, WQ, Y, int splitK); all three a16w16-family
// kernels populate the same GENERATE_A16W16_TUNE_LOOKUP map:
//   * split-barrier a16w16 (kids 4..9)      - ignores splitK
//   * a16w16_flatmm      (kids 100..115)    - ignores splitK
//   * a16w16_flatmm_splitk (kids 200..210)  - interprets splitK as literal KBatch
//
// splitk kids require D_C=fp32 (main kernel writes an fp32 workspace; the
// reduce kernel D_OUT is templated on Y.dtype() and chosen at launch time),
// so the dispatcher forces the <fp32_t> branch for kids >= 200 regardless of
// Y dtype. Both bf16 and fp32 Y are valid.

using OpusA16W16TuneKernel = OpusA16W16NoscaleKernel;

using OpusA16W16TuneMap = std::unordered_map<
    int,
    OpusA16W16TuneKernel>;

// Two specializations, each using its own lookup macro. The bf16 map omits
// splitk kids (their <bf16_t> instantiation doesn't exist; splitk main kernel
// hardcodes D_C=float). The fp32 map includes all a16w16-family kids; splitk
// kids appear there with <fp32_t> hardcoded as well, since the reduce kernel
// handles fp32 Y output by skipping the cast.
template <typename CDataType>
OpusA16W16TuneKernel opus_a16w16_tune_dispatch(int id);

template <>
OpusA16W16TuneKernel opus_a16w16_tune_dispatch<bf16_t>(int id)
{
  static const auto lookup = []
  {
    return OpusA16W16TuneMap{GENERATE_A16W16_TUNE_LOOKUP_BF16(bf16_t)};
  }();
  auto it = lookup.find(id);
  TORCH_CHECK(it != lookup.end(),
              "Kernel id " + std::to_string(id) + " not found in a16w16 bf16 tune lookup table");
  return it->second;
}

template <>
OpusA16W16TuneKernel opus_a16w16_tune_dispatch<fp32_t>(int id)
{
  static const auto lookup = []
  {
    return OpusA16W16TuneMap{GENERATE_A16W16_TUNE_LOOKUP_FP32(fp32_t)};
  }();
  auto it = lookup.find(id);
  TORCH_CHECK(it != lookup.end(),
              "Kernel id " + std::to_string(id) + " not found in a16w16 fp32 tune lookup table");
  return it->second;
}

// splitk kids live in [200, 300). Kept as a range so future splitk tiles don't
// require a dispatcher change.
static constexpr int OPUS_SPLITK_KID_MIN = 200;
static constexpr int OPUS_SPLITK_KID_MAX = 300;

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
    // splitk kids force <fp32_t> because traits static_assert D_C=float.
    // Y can be bf16 or fp32 -- the launcher dispatches the reduce kernel
    // on Y.dtype() at runtime.
    if (kernelId >= OPUS_SPLITK_KID_MIN && kernelId < OPUS_SPLITK_KID_MAX)
    {
      TORCH_CHECK(Y.dtype() == at::ScalarType::BFloat16
                  || Y.dtype() == at::ScalarType::Float,
                  "opus_gemm_a16w16_tune splitk kid requires bf16 or fp32 Y "
                  "(reduce kernel writes the correct dtype)");
      opus_a16w16_tune_dispatch<fp32_t>(kernelId)(XQ, WQ, Y, splitK);
    }
    else if (Y.dtype() == at::ScalarType::BFloat16)
    {
      opus_a16w16_tune_dispatch<bf16_t>(kernelId)(XQ, WQ, Y, splitK);
    }
    else if (Y.dtype() == at::ScalarType::Float)
    {
      opus_a16w16_tune_dispatch<fp32_t>(kernelId)(XQ, WQ, Y, splitK);
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
