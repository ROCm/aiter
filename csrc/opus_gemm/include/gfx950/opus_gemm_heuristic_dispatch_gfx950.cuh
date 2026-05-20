// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a16w16 family heuristic dispatcher (gfx950).
//
// The heuristic is the "no-tuned-CSV fallback" arm of opus_dispatch_a16w16:
// when a runtime (M,N,K) shape has no row in opus_gemm_lookup.h, we still
// need to pick *some* valid a16w16 kernel for it. This file defines that
// pick as a pure ``(M,N,K) -> kid`` mapping; the caller (see
// opus_gemm_arch_gfx950.cuh) then resolves the kid through
// opus_a16w16_tune_dispatch_gfx950<>() against the (gen_instances.py-
// emitted) tune lookup table.
//
// Why kid integers instead of launcher symbol names?
// ---------------------------------------------------
// The previous version of this file returned bare ``&opus_gemm_..._wgpcu1
// <fp32_t>`` symbols. That coupled the heuristic to specific .so symbol
// names, which is a real problem in the subset-compile world: if a build
// excludes the splitk-128 launcher (because the CSV doesn't ask for it
// and the heuristic doesn't either), but the .cuh still references the
// symbol, the link fails. By returning an integer kid here and routing
// through the tune lookup, the only invariant is "every kid this function
// can return must also be in the compiled subset S". That invariant is
// enforced at *codegen* time by csrc/opus_gemm/gen_instances.py
//   assert HEURISTIC_DEFAULT_KIDS.issubset(S)
// using the single source of truth in opus_gemm_common.py.
//
// Keep the integer kid returns in opus_a16w16_heuristic_kid_gfx950() below
// in sync with the HEURISTIC_DEFAULT_KIDS frozenset in
// csrc/opus_gemm/opus_gemm_common.py. The two are coupled by intent.
//
// gfx950-specific because the choices below were profiled on MI350's
// 256-CU / 160 KB LDS budget. Future archs will have their own
// opus_gemm_heuristic_dispatch_<arch>.cuh next to this one.
#pragma once

#include <optional>

#include "aiter_tensor.h"  // aiter_tensor_t (torch-free)
#include "../opus_gemm_common.cuh"
#include "opus_gemm_manifest.h"

// a16w16-family launcher signature (split-barrier, flatmm, flatmm_splitk):
// 3 tensors + std::optional<bias> + int splitK so all three populate the
// same GENERATE_A16W16_TUNE_LOOKUP table. Non-splitk launchers ignore
// splitK; the splitk launcher treats it as literal KBatch. bias is
// consumed by the split-barrier and splitk launchers; the flatmm launcher
// rejects any non-empty bias up front (HAS_BIAS=false on its warp-spec
// epilogue).
//
// Returns void (in-place on Y); the launchers used to return Y but
// nothing read the return value at any call site, and dropping the
// torch::Tensor return type lets the whole dispatch graph go
// torch-free.
//
// Plain function pointer (was: `std::function<void(...)>`). Every
// callable we ever store in this slot is one of the explicitly
// instantiated `xxx<bf16_t>` / `xxx<fp32_t>` launcher templates --
// no captures, no type erasure needed. Switching to a function
// pointer drops a heavyweight `std::function` template instantiation
// from the dispatcher TU's host pass and also avoids the per-call
// virtual-dispatch overhead that std::function pays for the type
// erasure we don't actually use.
using OpusA16W16NoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, std::optional<aiter_tensor_t>, int);


// Pure (M, N, K, has_bias) -> integer kid mapping. No reference to launcher
// symbols here -- the caller resolves the returned kid through
// opus_a16w16_tune_dispatch_gfx950<CDataType>(kid).
//
// IMPORTANT: every kid this function can return MUST also be in
// HEURISTIC_DEFAULT_KIDS in csrc/opus_gemm/opus_gemm_common.py, so
// the subset-compile codegen always includes them in S.
//
// `has_bias` matters because the persistent pipeline does not yet
// implement HAS_BIAS=true; when the user passes a non-empty bias the
// heuristic must stay on the bias-aware splitk family even if the M-bucket
// would otherwise return a persistent kid. Splitk kids 200/206/208 (+
// nooob mirrors) are all bias-aware (see opus_kid_supports_bias in
// opus_gemm.cu and BIAS_AWARE_KIDS in opus_gemm_common.py).
inline int opus_a16w16_heuristic_kid_gfx950(int M, int N, int K, bool has_bias = false)
{
  const bool split_barrier_ok =
      (N % 16 == 0) && (K % 64 == 0) && ((K / 64) % 2 == 0);

  if (M <= 4)
  {
    // Extremely skinny M: cc recommends (64,64,128) WG=1 for deep K.
    // kid 208 (oob) / 1208 (nooob): a16w16_flatmm_splitk_64x64x128_wgpcu1.
    if ((M % 64 == 0) && (N % 64 == 0) && (K % 128 == 0))
      return 1208;
    return 208;
  }
  if (M <= 64)
  {
    // Mid-skinny: cc-recommended medium-M kernel (64,32,128) WG=2.
    // kid 206 (oob) / 1206 (nooob): a16w16_flatmm_splitk_64x32x128_wgpcu2.
    if ((M % 64 == 0) && (N % 32 == 0) && (K % 128 == 0))
      return 1206;
    return 206;
  }
  if (M <= 128)
  {
    // Sweet spot: (64,64,64) WG=2.
    // kid 200 (oob) / 1200 (nooob): a16w16_flatmm_splitk_64x64x64_wgpcu2.
    if ((M % 64 == 0) && (N % 64 == 0) && (K % 64 == 0))
      return 1200;
    return 200;
  }
  // M > 128
  if (split_barrier_ok && !has_bias)
  {
    // Persistent (256, 256, 64) tile; CDataType-templated by caller.
    // kid 300 (oob) / 1300 (nooob). Persistent does not yet support bias --
    // when has_bias is true we fall through to the splitk path below.
    if ((M % 256 == 0) && (N % 256 == 0) && (K % 64 == 0))
      return 1300;
    return 300;
  }
  // M > 128 but split-barrier prerequisites failed (or bias requested) --
  // fall back to the splitk sweet-spot tile. Splitk supports bias.
  if ((M % 64 == 0) && (N % 64 == 0) && (K % 64 == 0))
    return 1200;
  return 200;
}
