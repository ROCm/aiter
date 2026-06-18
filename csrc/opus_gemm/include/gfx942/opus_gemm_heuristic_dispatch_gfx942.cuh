// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Coarse gfx942 a16w16 fallback heuristic. Exact tuned shapes should hit the
// generated CSV lookup first; this path only needs a sane family-level guess.
#pragma once

#include <optional>
#include <type_traits>

#include "aiter_tensor.h"
#include "../opus_gemm_common.cuh"

#ifndef OPUS_A16W16_NOSCALE_KERNEL_DEFINED
#define OPUS_A16W16_NOSCALE_KERNEL_DEFINED
using OpusA16W16NoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, std::optional<aiter_tensor_t>, int);
#endif

#define OPUS_GFX942_A16W16_DECL(NAME)                                                \
template <typename D_C>                                                              \
void NAME(aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,                      \
          std::optional<aiter_tensor_t>, int)

OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0);
OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_p1_256x64x64x64_2x2_16x16x16_0x0x0);
OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0);
OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0);
OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_splitk_em3en4_lds1_pgr2_256x128x96x128_2x2_16x16x16_0x0x0);
OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_splitk_legacy_bf16ws_512x128x128x64_2x4_16x16x16_0x0x0);
OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_wkc_512x16x16x64_1x1_16x16x16_0x0x0);
OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_wkc_512x16x32x32_1x1_16x16x16_0x0x0);
OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0);
OPUS_GFX942_A16W16_DECL(opus_gemm_gfx942_wkc_256x32x32x64_1x1_16x16x16_0x0x0);

#undef OPUS_GFX942_A16W16_DECL

namespace opus_gfx942_heuristic_detail
{

inline bool split_barrier_ok(int N, int K)
{
  const int loops = (K + 63) / 64;
  return (N % 16 == 0) && (K % 64 == 0) && (loops >= 2) && (loops % 2 == 0);
}

inline bool bf16ws_band(int M, int N, int K)
{
  return (K >= 4096) && (K % 64 == 0) && (M >= 104) && (M <= 608) &&
         (N == 256 || (N >= 512 && N <= 2048));
}

template <typename CDataType>
inline OpusA16W16NoscaleKernel dispatch_bf16(int M, int N, int K)
{
  const bool k64_ok = K % 64 == 0;
  const bool k32_ok = K % 32 == 0;
  const bool p1_ok = K % 128 == 0;
  const bool sb_ok = split_barrier_ok(N, K);

  if (K >= 1024 && k32_ok && N >= 1536 && M <= 32)
  {
    if (M <= 4 && N >= 4096)
      return opus_gemm_gfx942_wkc_512x16x16x64_1x1_16x16x16_0x0x0<CDataType>;
    return M <= 16
        ? opus_gemm_gfx942_wkc_512x16x32x32_1x1_16x16x16_0x0x0<CDataType>
        : opus_gemm_gfx942_wkc_256x32x32x64_1x1_16x16x16_0x0x0<CDataType>;
  }

  if (K >= 512 && k64_ok && (N <= 64 || (M <= 128 && N <= 1024) ||
                             (M <= 8 && N <= 1536)))
  {
    if (N <= 64 && M > 128)
      return opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0<CDataType>;
    if (N <= 256 || M <= 8 || (M <= 16 && N <= 800))
      return opus_gemm_gfx942_wkc_512x16x16x64_1x1_16x16x16_0x0x0<CDataType>;
    return opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0<CDataType>;
  }

  if (bf16ws_band(M, N, K))
    return opus_gemm_gfx942_splitk_legacy_bf16ws_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;

  if (N == 384 && K >= 4096)
  {
    if (M <= 128)
      return opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0<CDataType>;
    if (M <= 224)
      return opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0<fp32_t>;
    if (M >= 392 && M <= 512)
      return opus_gemm_gfx942_splitk_em3en4_lds1_pgr2_256x128x96x128_2x2_16x16x16_0x0x0<fp32_t>;
    return opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;
  }

  if (k64_ok && N >= 4096 && K <= 3200)
  {
    if (K <= 640 && M <= 128)
      return opus_gemm_gfx942_p1_256x64x64x64_2x2_16x16x16_0x0x0<CDataType>;
    return opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0<CDataType>;
  }

  if (sb_ok && M >= 128)
    return opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0<CDataType>;

  if (N <= 256 && p1_ok)
    return opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0<fp32_t>;

  return opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;
}

}  // namespace opus_gfx942_heuristic_detail

template <typename CDataType>
inline OpusA16W16NoscaleKernel opus_a16w16_heuristic_dispatch_gfx942(
    int M, int N, int K, int /*batch*/, bool has_bias = false)
{
  using namespace opus_gfx942_heuristic_detail;

  if constexpr (std::is_same_v<CDataType, bf16_t>)
  {
    if (!has_bias)
      return dispatch_bf16<CDataType>(M, N, K);
  }

  if (N <= 256 && K % 128 == 0)
    return opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0<fp32_t>;

  return opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;
}
