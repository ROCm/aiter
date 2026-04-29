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
#include "py_itfs_common.h"                         // bf16_t / fp32_t

#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>

namespace opus_gfx950_detail
{
struct IntTupleHash
{
    size_t operator()(const std::tuple<int, int, int> &t) const
    {
        auto h1 = std::hash<int>{}(std::get<0>(t));
        auto h2 = std::hash<int>{}(std::get<1>(t));
        auto h3 = std::hash<int>{}(std::get<2>(t));
        return h1 ^ h2 ^ h3;
    }
};

// (M, N, K) -> kernel<CDataType>. Populated from opus-private tuned CSV at
// JIT time by gen_instances.py --tune_file. Mirrors the IntTupleHash +
// unordered_map layout used by csrc/ck_gemm_a8w8/gemm_a8w8.cu.
using OpusA16W16RuntimeMap = std::unordered_map<
    std::tuple<int, int, int>,
    OpusA16W16NoscaleKernel,
    IntTupleHash>;

// id -> kernel<CDataType>. Same launcher signature as the runtime map; both
// the bf16 and fp32 specializations use the same alias.
using OpusA16W16TuneKernel = OpusA16W16NoscaleKernel;
using OpusA16W16TuneMap    = std::unordered_map<int, OpusA16W16TuneKernel>;
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
    static const auto lookup = []
    {
        return OpusA16W16RuntimeMap{GENERATE_OPUS_LOOKUP_TABLE_BF16(bf16_t)};
    }();
    auto it = lookup.find({M, N, K});
    if (it != lookup.end())
    {
        return it->second;
    }
    return opus_a16w16_heuristic_dispatch_gfx950<bf16_t>(M, N, K, batch);
}

template <>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx950<fp32_t>(int M, int N, int K, int batch)
{
    using namespace opus_gfx950_detail;
    static const auto lookup = []
    {
        return OpusA16W16RuntimeMap{GENERATE_OPUS_LOOKUP_TABLE_FP32(fp32_t)};
    }();
    auto it = lookup.find({M, N, K});
    if (it != lookup.end())
    {
        return it->second;
    }
    return opus_a16w16_heuristic_dispatch_gfx950<fp32_t>(M, N, K, batch);
}

// ── a16w16 tune dispatch (id-based, two specializations) ────────────────────
//
// The bf16 map omits splitk kids (their <bf16_t> instantiation doesn't exist;
// splitk main kernel hardcodes D_C=float). The fp32 map includes all
// a16w16-family kids; splitk kids appear there with <fp32_t> hardcoded as
// well, since the reduce kernel handles fp32 Y output by skipping the cast.

template <typename CDataType>
inline opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx950(int id);

template <>
inline opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx950<bf16_t>(int id)
{
    using namespace opus_gfx950_detail;
    static const auto lookup = []
    {
        return OpusA16W16TuneMap{GENERATE_A16W16_TUNE_LOOKUP_BF16(bf16_t)};
    }();
    auto it = lookup.find(id);
    TORCH_CHECK(it != lookup.end(),
                "Kernel id " + std::to_string(id) +
                " not found in a16w16 bf16 tune lookup table");
    return it->second;
}

template <>
inline opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx950<fp32_t>(int id)
{
    using namespace opus_gfx950_detail;
    static const auto lookup = []
    {
        return OpusA16W16TuneMap{GENERATE_A16W16_TUNE_LOOKUP_FP32(fp32_t)};
    }();
    auto it = lookup.find(id);
    TORCH_CHECK(it != lookup.end(),
                "Kernel id " + std::to_string(id) +
                " not found in a16w16 fp32 tune lookup table");
    return it->second;
}
