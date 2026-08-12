// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_arch_gfx950.cuh — gfx950-specific dispatch implementations.
//
// Provides:
//   * opus_dispatch_a16w16_gfx950<T>      — tuned (M,N,K) lookup → heuristic
//   * opus_a16w16_tune_dispatch_gfx950<T> — id-based tune dispatch
//
// This header is intended to be included exactly once, by opus_gemm.cu, where
// the arch routers in that TU select the per-arch entry. Other TUs (the
// launcher instances) must NOT include it -- they would each pull in the
// generated lookup macros (~70 KiB) for no gain.
//
// To add a new arch (e.g. gfx942):
//   1. Add OpusGfxArch::Gfx942 to opus_gemm_arch.cuh.
//   2. Create opus_gemm_arch_gfx942.cuh mirroring this file's shape; provide
//      the per-arch dispatch functions with whatever lookup / heuristic that
//      arch needs (it can reuse the same lookup macros if applicable, or
//      its own).
//   3. #include "opus_gemm_arch_gfx942.cuh" in opus_gemm.cu and add a
//      `case OpusGfxArch::Gfx942: ...` branch to each arch router there.
#pragma once

#include "../opus_gemm_arch.cuh"
#include "../opus_gemm_common.cuh"
#include "opus_gemm_heuristic_dispatch_gfx950.cuh"  // OpusA16W16NoscaleKernel + opus_a16w16_heuristic_kid_gfx950()
#include "opus_gemm_lookup.h"                       // GENERATE_OPUS_LOOKUP_TABLE_BF16 / FP32
#include "opus_gemm_a16w16_tune_lookup.h"           // GENERATE_A16W16_TUNE_LOOKUP_BF16 / FP32
#include "opus_gemm_manifest.h"                     // launcher symbols referenced by the lookup macros
#include "../opus_gemm_utils.cuh"                   // bf16_t / fp32_t (torch-free; py_itfs_common.h pulls full <torch/all.h>)

#include <algorithm>  // std::lower_bound
#include <cstddef>

namespace opus_gfx950_detail
{
// Sorted flat-array entries for the runtime (M, N, K) -> kernel lookup
// (was: std::unordered_map<std::tuple<int,int,int>,
// OpusA16W16NoscaleKernel, IntTupleHash>). The unordered_map version
// added ~1s of frontend / template instantiation per dispatcher TU
// because of the heavyweight std::function-valued hashtable templates;
// a flat array of POD entries plus std::lower_bound costs essentially
// nothing at parse time and matches the lookup at runtime in O(log N)
// over 339 entries.
// Nested {shape, func} aggregate matches the `{ {M, N, K}, &kernel }`
// initializer the codegen emits. Splitting shape into its own struct
// keeps the comparators small and gives gen_instances.py a stable
// brace pattern to target.
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

// Lex order on (M, N, K). Used both during sorting (gen_instances.py
// emits entries in lex order) and by std::lower_bound at lookup time.
constexpr bool entry_less(const OpusA16W16RuntimeEntry& a,
                          const OpusA16W16RuntimeEntry& b) noexcept
{
    if (a.key.M != b.key.M) return a.key.M < b.key.M;
    if (a.key.N != b.key.N) return a.key.N < b.key.N;
    return a.key.K < b.key.K;
}

constexpr bool entry_eq(const OpusA16W16RuntimeEntry& a,
                        const OpusA16W16RuntimeEntry& b) noexcept
{
    return a.key.M == b.key.M && a.key.N == b.key.N && a.key.K == b.key.K;
}

// id -> kernel<CDataType>, same flat-array layout. Sorted by id (the
// codegen always emits in ascending id order).
struct OpusA16W16TuneEntry
{
    int kid;
    OpusA16W16NoscaleKernel func;
};

constexpr bool tune_entry_less(const OpusA16W16TuneEntry& a,
                               const OpusA16W16TuneEntry& b) noexcept
{
    return a.kid < b.kid;
}

using OpusA16W16TuneKernel = OpusA16W16NoscaleKernel;
}  // namespace opus_gfx950_detail

// Splitk kid range. Kept in this header (rather than relying on the
// opus_gemm.cu copy in OPUS_SPLITK_KID_MIN/MAX) so the heuristic-fallback
// path below can route splitk kids to <fp32_t> tune_dispatch without a
// cross-TU dependency. The numbers must match opus_gemm.cu.
namespace opus_gfx950_detail
{
constexpr int kSplitkKidMin       = 200;
constexpr int kSplitkKidMax       = 300;
constexpr int kNooobKidOffset     = 1000;

constexpr bool kid_is_splitk(int kid) noexcept
{
    return (kid >= kSplitkKidMin && kid < kSplitkKidMax) ||
           (kid >= kSplitkKidMin + kNooobKidOffset &&
            kid < kSplitkKidMax + kNooobKidOffset);
}
}  // namespace opus_gfx950_detail

// ── a16w16 tune dispatch (id-based, two specializations) ────────────────────
//
// The bf16 table omits splitk kids (their <bf16_t> instantiation doesn't
// exist; splitk main kernel hardcodes D_C=float). The fp32 table includes
// all a16w16-family kids; splitk kids appear there with <fp32_t>
// hardcoded as well, since the reduce kernel handles fp32 Y output by
// skipping the cast.

template <typename CDataType>
inline opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx950(int id);

template <>
inline opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx950<bf16_t>(int id)
{
    using namespace opus_gfx950_detail;
    static constexpr OpusA16W16TuneEntry kTune[] = {
        GENERATE_A16W16_TUNE_LOOKUP_BF16(bf16_t)
    };
    constexpr size_t kSize = sizeof(kTune) / sizeof(kTune[0]);
    OpusA16W16TuneEntry needle{id, nullptr};
    auto it = std::lower_bound(kTune, kTune + kSize, needle, tune_entry_less);
    AITER_CHECK(it != kTune + kSize && it->kid == id,
                "Kernel id ", id,
                " not found in a16w16 bf16 tune lookup table");
    return it->func;
}

template <>
inline opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx950<fp32_t>(int id)
{
    using namespace opus_gfx950_detail;
    static constexpr OpusA16W16TuneEntry kTune[] = {
        GENERATE_A16W16_TUNE_LOOKUP_FP32(fp32_t)
    };
    constexpr size_t kSize = sizeof(kTune) / sizeof(kTune[0]);
    OpusA16W16TuneEntry needle{id, nullptr};
    auto it = std::lower_bound(kTune, kTune + kSize, needle, tune_entry_less);
    AITER_CHECK(it != kTune + kSize && it->kid == id,
                "Kernel id ", id,
                " not found in a16w16 fp32 tune lookup table");
    return it->func;
}

// ── a16w16 runtime dispatch (tuned lookup → heuristic fallback) ─────────────
//
// On miss the heuristic returns an integer kid; we re-dispatch through
// opus_a16w16_tune_dispatch_gfx950<>(). Splitk kids only have a <fp32_t>
// instantiation (their traits static_assert D_C=float; the reduce kernel
// templated on Y dtype handles bf16/fp32 output at launch time), so we
// force the <fp32_t> branch for those regardless of the dispatcher's
// CDataType template parameter.

template <typename CDataType>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx950(int M, int N, int K, int batch, bool has_bias = false);

template <>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx950<bf16_t>(int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx950_detail;
    static constexpr OpusA16W16RuntimeEntry kLookup[] = {
        GENERATE_OPUS_LOOKUP_TABLE_BF16(bf16_t)
    };
    constexpr size_t kSize = sizeof(kLookup) / sizeof(kLookup[0]);
    OpusA16W16RuntimeEntry needle{{M, N, K}, nullptr};
    auto it = std::lower_bound(kLookup, kLookup + kSize, needle, entry_less);
    if (it != kLookup + kSize && entry_eq(*it, needle))
    {
        return it->func;
    }
    (void)batch;  // heuristic does not currently use batch.
    // 4 GiB buffer-resource guard. The heuristic returns one of
    // HEURISTIC_DEFAULT_KIDS, all of which are legacy (non-4g_safe) and
    // build a single AMDGPU buffer-resource over the whole A/B/C tensors;
    // 32-bit num_records wraps when any A/B/C bytes exceed UINT32_MAX,
    // producing silent OOB. Refuse fallback for >4 GiB shapes -- the
    // caller must register a tuned CSV entry mapping the shape to a
    // 4g_safe kid (5000-series / 6000-series).
    constexpr uint64_t U32_MAX_BYTES = (1ULL << 32) - 1;
    const uint64_t a_bytes = (uint64_t)M * (uint64_t)K * sizeof(bf16_t);
    const uint64_t b_bytes = (uint64_t)N * (uint64_t)K * sizeof(bf16_t);
    const uint64_t c_bytes = (uint64_t)M * (uint64_t)N * sizeof(bf16_t);
    AITER_CHECK(a_bytes <= U32_MAX_BYTES && b_bytes <= U32_MAX_BYTES
                    && c_bytes <= U32_MAX_BYTES,
                "opus a16w16 heuristic fallback refuses >4 GiB shape (M=",
                M, " N=", N, " K=", K,
                "): legacy kids wrap buffer-resource num_records. "
                "Add a tuned CSV entry mapping this shape to a 4g_safe kid "
                "(5000-series for split-barrier / 6000-series for mono_tile).");
    // has_bias forces the heuristic to skip persistent kids (which
    // do not yet implement HAS_BIAS=true) and return a splitk kid instead.
    const int kid = opus_a16w16_heuristic_kid_gfx950(M, N, K, has_bias);
    if (kid_is_splitk(kid))
        return opus_a16w16_tune_dispatch_gfx950<fp32_t>(kid);
    return opus_a16w16_tune_dispatch_gfx950<bf16_t>(kid);
}

template <>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx950<fp32_t>(int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx950_detail;
    static constexpr OpusA16W16RuntimeEntry kLookup[] = {
        GENERATE_OPUS_LOOKUP_TABLE_FP32(fp32_t)
    };
    constexpr size_t kSize = sizeof(kLookup) / sizeof(kLookup[0]);
    OpusA16W16RuntimeEntry needle{{M, N, K}, nullptr};
    auto it = std::lower_bound(kLookup, kLookup + kSize, needle, entry_less);
    if (it != kLookup + kSize && entry_eq(*it, needle))
    {
        return it->func;
    }
    (void)batch;
    // 4 GiB buffer-resource guard (see <bf16_t> overload for rationale).
    // C is fp32 here so the bound is 4 bytes/element.
    constexpr uint64_t U32_MAX_BYTES = (1ULL << 32) - 1;
    const uint64_t a_bytes = (uint64_t)M * (uint64_t)K * sizeof(bf16_t);
    const uint64_t b_bytes = (uint64_t)N * (uint64_t)K * sizeof(bf16_t);
    const uint64_t c_bytes = (uint64_t)M * (uint64_t)N * sizeof(fp32_t);
    AITER_CHECK(a_bytes <= U32_MAX_BYTES && b_bytes <= U32_MAX_BYTES
                    && c_bytes <= U32_MAX_BYTES,
                "opus a16w16 heuristic fallback refuses >4 GiB shape (M=",
                M, " N=", N, " K=", K,
                "): legacy kids wrap buffer-resource num_records. "
                "Add a tuned CSV entry mapping this shape to a 4g_safe kid "
                "(5000-series for split-barrier / 6000-series for mono_tile).");
    const int kid = opus_a16w16_heuristic_kid_gfx950(M, N, K, has_bias);
    // splitk kids only have <fp32_t> in the tune table; non-splitk kids
    // have an <fp32_t> entry in the fp32 lookup, so the branch is uniform.
    return opus_a16w16_tune_dispatch_gfx950<fp32_t>(kid);
}
