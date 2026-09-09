// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 a16w16 shape-heuristic: (M, N, K, has_bias) -> kid. Pure integer
// mapping (no launcher symbols) so it can be included by the dispatcher TU
// without dragging in the lookup macros.
//
// All gfx1250 kids are cluster/TDM split-K (workspace + reduce). The kernel
// requires M % B_M == 0 and N % B_N == 0 (ragged M/N is not supported; ragged
// K is, via the TDM k_extent clamp). The heuristic therefore picks the largest
// tile from the kid set whose B_M divides M and B_N divides N, preferring the
// B_M=16 "tileN" family for small M and the "tileM" family for larger M.
//
// MUST stay in sync with opus_gemm_common.py :: gfx1250_kernels_list and
// HEURISTIC_DEFAULT_KIDS_GFX1250.
#pragma once

#include <optional>

#include "aiter_tensor.h"  // aiter_tensor_t (torch-free)

// Shared flat-array dispatch POD types + comparators for gfx1250 (mirrors the
// gfx950 set). gen_instances.py emits the tune / (M,N,K) lookup tables as
// arrays of these; std::lower_bound does the O(log N) runtime match.
namespace opus_gfx1250_detail
{

// a16w16-family launcher signature for gfx1250: 3 tensors + workspace +
// std::optional<bias> + int splitK. Different from gfx950 (no workspace).
using OpusA16W16NoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, aiter_tensor_t &, std::optional<aiter_tensor_t>, int);

struct OpusA16W16Shape
{
    int M;
    int N;
    int K;
};

struct OpusA16W16RuntimeEntry
{
    OpusA16W16Shape key;
    OpusA16W16NoscaleKernel func;
};

// Comparators are templated on the entry type rather than written once per
// table: the entry types differ only in what they carry NEXT to the key (a
// workspace-carrying function pointer, a workspace-free one, ...), and every
// table is keyed the same way. Explicit template args at the call sites, since
// std::lower_bound has no target type to deduce them from.
//
// Lex order on (M, N, K). Used both during sorting (gen_instances.py emits
// entries in lex order) and by std::lower_bound at lookup time.
template <typename Entry>
constexpr bool shape_entry_less(const Entry& a, const Entry& b) noexcept
{
    if (a.key.M != b.key.M) return a.key.M < b.key.M;
    if (a.key.N != b.key.N) return a.key.N < b.key.N;
    return a.key.K < b.key.K;
}

template <typename Entry>
constexpr bool shape_entry_eq(const Entry& a, const Entry& b) noexcept
{
    return a.key.M == b.key.M && a.key.N == b.key.N && a.key.K == b.key.K;
}

template <typename Entry>
constexpr bool kid_entry_less(const Entry& a, const Entry& b) noexcept
{
    return a.kid < b.kid;
}

// id -> kernel<CDataType>, same flat-array layout. Sorted by id (the codegen
// always emits in ascending id order).
struct OpusA16W16TuneEntry
{
    int kid;
    OpusA16W16NoscaleKernel func;
};

using OpusA16W16TuneKernel = OpusA16W16NoscaleKernel;

// ── pre-compiled .co kids (a16w16_4wave_co) ─────────────────────────────────
// A second function-pointer type, and deliberately so: this family has no
// split-K, no partial buffer and no reduce kernel, so it has nothing to put in
// a workspace, and its launcher is the ordinary 5-arg a16w16 signature (the
// same one gfx950/gfx942 use) rather than the 6-arg gfx1250 one above.
//
// The two could be collapsed by making the workspace argument
// std::optional<aiter_tensor_t> everywhere, but that would ERASE exactly the
// fact worth keeping: with "needs a workspace" in the type, the production
// dispatch in opus_gemm.cu can consult the .co table first and skip the
// hipMalloc / hipDeviceSynchronize / hipFree it has to do for every split-K
// kid. A merged type cannot tell, from a function pointer alone, whether the
// buffer is going to be read.
//
// A split-K .co variant would use OpusA16W16NoscaleKernel and these types go
// unused for it; nothing here has to change.
using OpusA16W16CoKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &, std::optional<aiter_tensor_t>, int);

struct OpusA16W16CoTuneEntry
{
    int kid;
    OpusA16W16CoKernel func;
};

struct OpusA16W16CoRuntimeEntry
{
    OpusA16W16Shape key;
    OpusA16W16CoKernel func;
};
}  // namespace opus_gfx1250_detail

// Kid map (B_K=128 chosen here; tuner explores B_K 256/512 + the P/wg space).
// Tiles whose per-TDM direct-copy request count (rows*B_K*2/256) hits the 256
// SIMD-pair limit on some operand are NOT generated (e.g. 32x256x128) so the
// heuristic must not return them. All returned kids are no-cluster prefetch-3.
//   tileN (B_M=16): 20000=16x32, 20003=16x64, 20004=16x128
//   tileM (B_M=32): 20005=32x32, 20006=32x64, 20007=32x128
// (One P=3 kid per tile in the contiguous plain band [20000,20100).)
// MUST stay in sync with opus_gemm_common.py :: gfx1250_kernels_list (the plain
// kids are assigned contiguously from 20000 in _GFX1250_CTDM_TILES order).
inline int opus_a16w16_heuristic_kid_gfx1250(int M, int N, int K, bool has_bias)
{
    (void)K;
    (void)has_bias;  // bias is folded by the reduce kernel for every kid.

    // M >= 32 (and M % 32 == 0) -> tileM (B_M=32); widest B_N that divides N.
    // (32x256 is unavailable -- per-TDM B req = 256 hits the direct-copy limit;
    // fall through to the B_M=16 tileN family for N % 256 == 0.)
    if (M % 32 == 0)
    {
        if (N % 128 == 0) return 20007;  // 32x128x128
        if (N % 64 == 0)  return 20006;  // 32x64x128
        if (N % 32 == 0)  return 20005;  // 32x32x128
    }

    // Small M (or N not tileM-friendly) -> tileN family (B_M=16). Ragged M/N is
    // handled by the TDM row/col clamp + padded workspace, so the smallest
    // 16x32 tile is always a valid fallback.
    if (N % 128 == 0) return 20004;  // 16x128x128
    if (N % 64 == 0)  return 20003;  // 16x64x128
    return 20000;                    // 16x32x128
}
