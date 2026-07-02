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
// Returns integer kids, not launcher symbol names: naming .so symbols directly
// would break the link in subset-compile builds that exclude a launcher the
// .cuh still references. Routing kids through the tune lookup reduces the
// invariant to "every kid returned here is in the compiled subset S", enforced
// at codegen time by gen_instances.py (assert HEURISTIC_DEFAULT_KIDS.issubset(S)).
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
// 3 tensors + std::optional<bias> + int splitK so all three populate the same
// GENERATE_A16W16_TUNE_LOOKUP table. Non-splitk launchers ignore splitK; the
// splitk launcher treats it as literal KBatch. bias is consumed by split-barrier
// and splitk launchers; flatmm rejects any non-empty bias (HAS_BIAS=false).
// Returns void (in-place on Y) to keep the dispatch graph torch-free. Plain
// function pointer (not std::function): every stored callable is an explicitly
// instantiated launcher template (no captures), so type erasure is unneeded and
// dropping std::function saves a heavy instantiation + per-call indirection.
using OpusA16W16NoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, std::optional<aiter_tensor_t>, int);


// Pure (M, N, K, has_bias) -> integer kid mapping; caller resolves the kid
// through opus_a16w16_tune_dispatch_gfx950<CDataType>(kid).
//
// IMPORTANT: every kid this function can return MUST also be in
// HEURISTIC_DEFAULT_KIDS in csrc/opus_gemm/opus_gemm_common.py, so
// the subset-compile codegen always includes them in S.
//
// has_bias matters because the persistent pipeline does not yet implement
// HAS_BIAS=true; with a non-empty bias the heuristic stays on the bias-aware
// splitk family (kids 200/206/208 + nooob mirrors; see BIAS_AWARE_KIDS in
// opus_gemm_common.py) even where the M-bucket would pick a persistent kid.
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
