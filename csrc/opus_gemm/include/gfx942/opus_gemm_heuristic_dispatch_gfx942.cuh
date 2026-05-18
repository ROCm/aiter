// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w16 family heuristic dispatcher (gfx942).
//
// Currently only one kernel is available (kid 6: 512x128x128x64, MFMA 16x16x16).
// All shapes fall through to this kernel. As more gfx942 kernels are added,
// extend the heuristic tree following the gfx950 pattern.
#pragma once

#include <optional>

#include "aiter_tensor.h"
#include "../opus_gemm_common.cuh"

// -- gfx942 launcher forward declarations ------------------------------------
// Add new gfx942 kernel declarations here as they land.

// kid 6: BS=512, B_M=128, B_N=128, B_K=64, T_M=2, T_N=4, MFMA=16x16x16
template <typename D_C>
void
opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0(
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

template <typename CDataType>
inline OpusA16W16NoscaleKernel opus_a16w16_heuristic_dispatch_gfx942(
    int M, int N, int K, int /*batch*/)
{
  // Only one kernel today; return it for all shapes.
  return opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0<CDataType>;
}
