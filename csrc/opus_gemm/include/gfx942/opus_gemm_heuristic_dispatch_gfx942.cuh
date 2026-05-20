// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w16 family heuristic dispatcher (gfx942).
//
// Available kernels:
//   * kid 6:   split-barrier a16w16 (512x128x128x64, MFMA 16x16x16) - no split-K
//   * kid 200: splitk a16w16       (512x128x128x64, MFMA 16x16x16) - independent reduce
//   * kid 201: splitk_fused a16w16 (512x128x128x64, MFMA 16x16x16) - fused reduce
#pragma once

#include <optional>

#include "aiter_tensor.h"
#include "../opus_gemm_common.cuh"

// -- gfx942 launcher forward declarations ------------------------------------
// Add new gfx942 kernel declarations here as they land.

// kid 6: split-barrier BS=512, B_M=128, B_N=128, B_K=64, T_M=2, T_N=4, MFMA=16x16x16
template <typename D_C>
void
opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 200: splitk BS=512, B_M=128, B_N=128, B_K=64, T_M=2, T_N=4, MFMA=16x16x16
template <typename D_C>
void
opus_gemm_gfx942_splitk_512x128x128x64_2x4_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 201: splitk_fused BS=512, B_M=128, B_N=128, B_K=64, T_M=2, T_N=4, MFMA=16x16x16
template <typename D_C>
void
opus_gemm_gfx942_splitk_fused_512x128x128x64_2x4_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// -- a16w16 launcher signature (shared with gfx950) -------------------------
#ifndef OPUS_A16W16_NOSCALE_KERNEL_DEFINED
#define OPUS_A16W16_NOSCALE_KERNEL_DEFINED
using OpusA16W16NoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, std::optional<aiter_tensor_t>, int);
#endif

// Single template body shared by both bf16 and fp32 specializations.
//
// Available gfx942 kernels (all share 512x128x128x64 tile, MFMA 16x16x16):
//   kid 6:   split-barrier a16w16 (no split-K overhead, lower latency for
//            large shapes). Requires N%16==0, K%64==0, ceil(K/64) even.
//   kid 200: splitk independent reduce (flexible alignment via host-side
//            auto-clamp; wins small-M where K-parallelism helps).
//   kid 201: splitk fused reduce (lower launch overhead than kid 200
//            but higher per-split cost; generally slower than 200).
//
// Strategy: prefer split-barrier (kid 6) when alignment constraints are
// met and M is large enough that the tile fills well. Fall back to splitk
// (kid 200) for small M or when alignment prevents split-barrier.
template <typename CDataType>
inline OpusA16W16NoscaleKernel opus_a16w16_heuristic_dispatch_gfx942(
    int M, int N, int K, int /*batch*/)
{
  const int loops = (K + 63) / 64;  // ceil_div(K, B_K=64)
  const bool split_barrier_ok =
      (N % 16 == 0) && (K % 64 == 0) && (loops >= 2) && (loops % 2 == 0);

  if (M <= 64)
  {
    // Small M: splitk benefits from K-parallelism.
    return opus_gemm_gfx942_splitk_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;
  }

  // M > 64: split-barrier avoids reduce-kernel overhead.
  if (split_barrier_ok)
  {
    return opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0<CDataType>;
  }

  // Alignment prevents split-barrier; fall back to splitk.
  return opus_gemm_gfx942_splitk_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;
}
