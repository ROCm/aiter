// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w16 family heuristic dispatcher (gfx942). See opus_gemm_common.py for
// the full kid catalogue (50000 / 50001-3 / 50011 / 50200-4 / 50211 / 50300).
#pragma once

#include <optional>
#include <type_traits>

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
opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 202: splitk BS=256, B_M=128, B_N=64, B_K=64, T_M=2, T_N=2, MFMA=16x16x16
template <typename D_C>
void
opus_gemm_gfx942_splitk_legacy_256x128x64x64_2x2_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 201: splitk BS=256, B_M=64, B_N=64, B_K=64, T_M=2, T_N=2, MFMA=16x16x16 (E_M=1)
template <typename D_C>
void
opus_gemm_gfx942_splitk_256x64x64x64_2x2_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 211: splitk_p1 BS=256, B_M=64, B_N=64, B_K=64, T_M=2, T_N=2, MFMA=16x16x16 (E_M=1)
// Same tile as 201 but K-dbuf depth=2 + V-dbuf + deterministic reduce; 2 wg/CU TLP target.
template <typename D_C>
void
opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 11: nosplitK p1 (a16w16_kbuf2v) BS=256, B_M=64, B_N=64, B_K=64 (E_M=1) -- 50011
template <typename D_C>
void
opus_gemm_gfx942_p1_256x64x64x64_2x2_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 2: nosplitK legacy (a16w16_kbuf1) BS=256, B_M=128, B_N=64, B_K=64 -- 50002
template <typename D_C>
void
opus_gemm_gfx942_legacy_256x128x64x64_2x2_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 3: nosplitK p1_bk128 (a16w16_kbuf2v_bk128) BS=256, B_M=64, B_N=64, B_K=128 -- 50003
template <typename D_C>
void
opus_gemm_gfx942_p1_bk128_256x64x64x128_2x2_16x16x16_0x0x0(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);

// kid 1: nosplitK w3 (a16w16_kbuf3) BS=256, B_M=64, B_N=64, B_K=64 -- 50001
template <typename D_C>
void
opus_gemm_gfx942_w3_256x64x64x64_2x2_16x16x16_0x0x0(
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

// Thread-local CU count cache (avoid hipGetDeviceProperties per dispatch).
inline int opus_gfx942_cu_num_cached()
{
  thread_local int cached = -1;
  if (cached < 0)
  {
    int dev = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) == hipSuccess &&
        hipGetDeviceProperties(&prop, dev) == hipSuccess)
    {
      cached = prop.multiProcessorCount;
    }
    if (cached <= 0) cached = 64;  // safe lower-bound for any gfx942 SKU
  }
  return cached;
}

// Strategy: tile coverage vs CU count. small_for_128 -> splitk (pick tile by
// N); else split-barrier kid 50000 when aligned, splitk 128x128 fallback.
template <typename CDataType>
inline OpusA16W16NoscaleKernel opus_a16w16_heuristic_dispatch_gfx942(
    int M, int N, int K, int /*batch*/, bool /*has_bias*/ = false)
{
  const int loops = (K + 63) / 64;  // ceil_div(K, B_K=64)
  const bool split_barrier_ok =
      (N % 16 == 0) && (K % 64 == 0) && (loops >= 2) && (loops % 2 == 0);

  const int cu_num    = opus_gfx942_cu_num_cached();
  const int tiles_128 = ((M + 127) / 128) * ((N + 127) / 128);
  const bool small_for_128 = tiles_128 < 2 * cu_num;

  if (small_for_128 || M <= 64)
  {
    // small-for-128: splitk by default; carve out nosplitK wins from tuned_v2.csv.
    // p1_parity_ok: K%128==0 (loops/split parity). ns_align_ok: K%64==0 + (K/64) even.
    const bool p1_parity_ok = (K % 128 == 0);
    const bool ns_align_ok = (K % 64 == 0) && ((K / 64) % 2 == 0);

    if constexpr (std::is_same_v<CDataType, bf16_t>) {
      const int grid_64 = ((M + 63) / 64) * ((N + 63) / 64);
      if (N <= 64 && ns_align_ok && grid_64 >= cu_num) {
        // M>=12288 -> legacy 128x64; else P1 64x64 (tuned_v2.csv).
        if (M >= 12288) {
          return opus_gemm_gfx942_legacy_256x128x64x64_2x2_16x16x16_0x0x0<CDataType>;
        }
        return opus_gemm_gfx942_p1_256x64x64x64_2x2_16x16x16_0x0x0<CDataType>;
      }
      if (N <= 128 && M < 6400 && M >= 2560 && ns_align_ok && grid_64 >= cu_num) {
        return opus_gemm_gfx942_p1_256x64x64x64_2x2_16x16x16_0x0x0<CDataType>;
      }
      // N>=128 + large-M: SB 50000 (128x128) wins even when tiles_128<2*cu.
      if (N >= 128 && M >= 4096 && split_barrier_ok) {
        return opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0<CDataType>;
      }
    }
    if (N <= 256 && p1_parity_ok) {
      return opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0<fp32_t>;
    }
    if (N <= 64) {
      return opus_gemm_gfx942_splitk_256x64x64x64_2x2_16x16x16_0x0x0<fp32_t>;
    }
    if (N <= 128)
    {
      return opus_gemm_gfx942_splitk_legacy_256x128x64x64_2x2_16x16x16_0x0x0<fp32_t>;
    }
    return opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;
  }

  // Large problem: split-barrier avoids reduce-kernel overhead.
  if (split_barrier_ok)
  {
    return opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0<CDataType>;
  }

  // Alignment prevents split-barrier; fall back to splitk.
  return opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0<fp32_t>;
}

// NOTE: splitK auto-pick lives inside each gfx942 splitk launcher itself (codegen'd from
// gen_instances.py: when splitK<=0 the lau...
