// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w16 family heuristic dispatcher (gfx950).
//
// gfx950-specific because the kernel symbols it returns are gfx950 tile
// sizes (e.g. 256x256x64 split-barrier, splitk 64x64x64 WG=2). Future archs
// will have their own opus_gemm_heuristic_dispatch_<arch>.cuh next to this
// one; the public OpusA16W16NoscaleKernel launcher signature stays
// arch-agnostic so the arch router in opus_gemm.cu can dispatch without
// caring about the per-arch tile choice.
//
// Split out of opus_gemm.cu so the dispatcher tree (M-bucket -> kernel)
// has its own home and can be edited without touching the
// opus_gemm()/opus_gemm_a16w16_tune() entry points. The runtime
// (M,N,K)->kernel lookup map (driven by the tuned CSV) and the
// torch-extension entry points stay in opus_gemm.cu; only the fall-
// through heuristic and the OpusA16W16NoscaleKernel signature live
// here.
//
// Mirrors the ck_gemm_a8w8 pattern (see csrc/ck_gemm_a8w8/gemm_a8w8.cu
// rowwise_dispatch): first consult a compile-time (M,N,K)->kernel table
// baked in from the opus-private tuned CSV via gen_instances.py
// --tune_file, then fall through to this hand-written heuristic if-else
// tree. The heuristic guarantees *some* valid kernel for every shape;
// its choice is deliberately conservative (favor splitk for small M
// because its host-side auto-clamp makes it alignment-tolerant, and
// the traditional a16w16 tile for M>128 where split-barrier pipelines
// still win throughput).
//
// The template parameter CDataType refers to the *accumulator / dispatch*
// type seen by the id-based tune lookup tables, NOT the user-visible Y
// dtype. Concretely:
//   * Traditional a16w16 kid 4..9 has both <bf16_t> and <fp32_t>
//     instantiations; CDataType == Y dtype.
//   * Splitk kid 200..210 only exists in <fp32_t> form (traits
//     static_assert D_C=float). Y can be bf16 or fp32 -- the reduce
//     kernel (splitk_reduce_kernel) is templated on D_OUT and chosen at
//     launch time based on Y.dtype(), so the same <fp32_t> main-kernel
//     instantiation feeds both output dtypes.
//
// Therefore:
//   * Y == bf16: any kid; splitk instances must be specialized on
//     <fp32_t> despite Y being bf16 (this is exactly what
//     opus_gemm_a16w16_tune does -- see OPUS_SPLITK_KID_MIN routing).
//   * Y == fp32: any kid; splitk instances are again specialized on
//     <fp32_t>, and the reduce kernel writes fp32 directly without
//     casting.
#pragma once

#include <functional>
#include <optional>

#include "aiter_tensor.h"  // aiter_tensor_t (torch-free)
#include "../opus_gemm_common.cuh"
#include "opus_gemm_manifest.h"

// a16w16-family launcher signature (split-barrier, flatmm, flatmm_splitk):
// 3 tensors + std::optional<bias> + int splitK so all three populate the
// same GENERATE_A16W16_TUNE_LOOKUP map. Non-splitk launchers ignore splitK;
// the splitk launcher treats it as literal KBatch. bias is consumed by the
// split-barrier and splitk launchers; the flatmm launcher rejects any
// non-empty bias up front (HAS_BIAS=false on its warp-spec epilogue).
//
// Returns void (in-place on Y); the launchers used to return Y but
// nothing read the return value at any call site, and dropping the
// torch::Tensor return type lets the whole dispatch graph go
// torch-free.
using OpusA16W16NoscaleKernel = std::function<
    void(aiter_tensor_t &, aiter_tensor_t &,
         aiter_tensor_t &, std::optional<aiter_tensor_t>, int)>;

// Single template body shared by both bf16 and fp32 specializations.
// All splitk kids are forced to <fp32_t> (their main kernel only has the
// fp32_t instantiation; the reduce kernel D_OUT is selected at launch
// time based on Y.dtype()). split-barrier kid 9 follows CDataType.
template <typename CDataType>
inline OpusA16W16NoscaleKernel opus_a16w16_heuristic_dispatch_gfx950(
    int M, int N, int K, int /*batch*/)
{
  const bool split_barrier_ok =
      (N % 16 == 0) && (K % 64 == 0) && ((K / 64) % 2 == 0);

  if (M <= 4)
  {
    // Extremely skinny M: cc recommends (64,64,128) WG=1 for deep K.
    return opus_gemm_flatmm_splitk_256x64x64x128_2x1_16x16x32_0x0x0_wgpcu1<fp32_t>;
  }
  if (M <= 64)
  {
    // Mid-skinny: cc-recommended medium-M kernel (64,32,128) WG=2.
    return opus_gemm_flatmm_splitk_256x64x32x128_2x1_16x16x32_0x0x0_wgpcu2<fp32_t>;
  }
  if (M <= 128)
  {
    // Sweet spot: (64,64,64) WG=2.
    return opus_gemm_flatmm_splitk_256x64x64x64_2x1_16x16x32_0x0x0_wgpcu2<fp32_t>;
  }
  // M > 128
  if (split_barrier_ok)
  {
    return opus_gemm_512x256x256x64_2x4_16x16x32_0x0x0<CDataType>;
  }
  // Non-aligned large shape: splitk kid 200 handles any (M, N, K) because
  // mask_va_tail + reduce-tail cover arbitrary N/K.
  return opus_gemm_flatmm_splitk_256x64x64x64_2x1_16x16x32_0x0x0_wgpcu2<fp32_t>;
}
