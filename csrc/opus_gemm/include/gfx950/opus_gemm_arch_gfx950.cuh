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
#include "opus_gemm_heuristic_dispatch_gfx950.cuh"  // OpusA16W16NoscaleKernel + opus_a16w16_heuristic_dispatch_gfx950<>
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

// ── a16w16 runtime dispatch (tuned lookup → heuristic fallback) ─────────────

template <typename CDataType>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx950(int M, int N, int K, int batch);

template <>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx950<bf16_t>(int M, int N, int K, int batch)
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
    return opus_a16w16_heuristic_dispatch_gfx950<bf16_t>(M, N, K, batch);
}

template <>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx950<fp32_t>(int M, int N, int K, int batch)
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
    return opus_a16w16_heuristic_dispatch_gfx950<fp32_t>(M, N, K, batch);
}

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
