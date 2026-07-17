// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_arch_gfx1250.cuh -- gfx1250-specific dispatch.
//
// Wires both call paths for the cluster/TDM split-K (workspace + reduce) kids:
//   * opus_a16w16_tune_dispatch_gfx1250<T>  -- id-based (explicit kernelId)
//   * opus_dispatch_a16w16_gfx1250<T>       -- tuned (M,N,K) lookup -> heuristic
//
// Every gfx1250 kid is a split-K kid whose main kernel writes an fp32 workspace
// (output_dtypes = ["fp32_t"]); the reduce kernel casts the partials to the
// runtime Y dtype (bf16/fp32) and folds bias. So all dispatch resolves through
// the <fp32_t> tune table -- the <bf16_t> specializations exist only to satisfy
// the shared arch-router template instantiation and are never invoked for
// gfx1250 (opus_gemm.cu forces <fp32_t> for split-K kids).
//
// Included exactly once, by opus_gemm.cu, AFTER gfx950's arch header so the
// shared flat-array helpers in opus_gfx950_detail are visible (reused here).
#pragma once

#include "../opus_gemm_arch.cuh"
#include "../opus_gemm_common.cuh"
#include "gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh"   // OpusA16W16NoscaleKernel + opus_gfx950_detail::*
#include "opus_gemm_heuristic_dispatch_gfx1250.cuh"          // opus_a16w16_heuristic_kid_gfx1250
#include "opus_gemm_lookup.h"                                // GENERATE_OPUS_LOOKUP_TABLE_FP32
#include "opus_gemm_a16w16_tune_lookup.h"                    // GENERATE_A16W16_TUNE_LOOKUP_FP32
#include "opus_gemm_manifest.h"                              // launcher symbols
#include "../opus_gemm_utils.cuh"                            // bf16_t / fp32_t

#include <algorithm>  // std::lower_bound
#include <cstddef>

// ── a16w16 tune dispatch (id-based) ─────────────────────────────────────────
// Only the <fp32_t> table is populated (gfx1250 kids are fp32-only split-K).
// The <bf16_t> specialization is defensive: it never carries a table (which
// would be an empty array in a gfx1250-only build) and is never called.

template <typename CDataType>
inline opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx1250(int id);

template <>
inline opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx1250<fp32_t>(int id)
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
                " not found in a16w16 fp32 tune lookup table (gfx1250)");
    return it->func;
}

template <>
inline opus_gfx950_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx1250<bf16_t>(int id)
{
    // gfx1250 split-K kids are emitted <fp32_t> only; the reduce kernel handles
    // bf16 Y output. opus_gemm.cu always routes split-K kids through <fp32_t>,
    // so this is unreachable -- but it must compile (no empty-array table).
    AITER_CHECK(false,
                "opus_gemm gfx1250: a16w16 <bf16_t> tune dispatch is not used "
                "(split-K kids are fp32-workspace; bf16 Y is produced by the "
                "reduce kernel). kid=", id);
    return nullptr;
}

// ── a16w16 runtime dispatch (tuned lookup -> heuristic fallback) ────────────
// Both dtype specializations route split-K kids through the <fp32_t> tune
// table (the launcher's reduce kernel produces the requested Y dtype).

namespace opus_gfx1250_detail
{
inline void check_shape_4g(int M, int N, int K, size_t c_elem_bytes)
{
    // 4 GiB buffer-resource guard: the launcher builds 32-bit-bounded gmem
    // descriptors over A/B/C, so >4 GiB tensors wrap num_records -> silent OOB.
    constexpr uint64_t U32_MAX_BYTES = (1ULL << 32) - 1;
    const uint64_t a_bytes = (uint64_t)M * (uint64_t)K * sizeof(bf16_t);
    const uint64_t b_bytes = (uint64_t)N * (uint64_t)K * sizeof(bf16_t);
    const uint64_t c_bytes = (uint64_t)M * (uint64_t)N * (uint64_t)c_elem_bytes;
    AITER_CHECK(a_bytes <= U32_MAX_BYTES && b_bytes <= U32_MAX_BYTES
                    && c_bytes <= U32_MAX_BYTES,
                "opus_gemm gfx1250: a16w16 heuristic refuses >4 GiB shape (M=",
                M, " N=", N, " K=", K, "): launcher gmem descriptors are 32-bit.");
}
}  // namespace opus_gfx1250_detail

template <typename CDataType>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx1250(int M, int N, int K, int batch, bool has_bias = false);

template <>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx1250<bf16_t>(int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx950_detail;
    static constexpr OpusA16W16RuntimeEntry kLookup[] = {
        GENERATE_OPUS_LOOKUP_TABLE_BF16(bf16_t)
    };
    constexpr size_t kSize = sizeof(kLookup) / sizeof(kLookup[0]);
    OpusA16W16RuntimeEntry needle{{M, N, K}, nullptr};
    auto it = std::lower_bound(kLookup, kLookup + kSize, needle, entry_less);
    if (it != kLookup + kSize && entry_eq(*it, needle))
        return it->func;
    (void)batch;
    opus_gfx1250_detail::check_shape_4g(M, N, K, sizeof(bf16_t));
    const int kid = opus_a16w16_heuristic_kid_gfx1250(M, N, K, has_bias);
    return opus_a16w16_tune_dispatch_gfx1250<fp32_t>(kid);
}

template <>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx1250<fp32_t>(int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx950_detail;
    static constexpr OpusA16W16RuntimeEntry kLookup[] = {
        GENERATE_OPUS_LOOKUP_TABLE_FP32(fp32_t)
    };
    constexpr size_t kSize = sizeof(kLookup) / sizeof(kLookup[0]);
    OpusA16W16RuntimeEntry needle{{M, N, K}, nullptr};
    auto it = std::lower_bound(kLookup, kLookup + kSize, needle, entry_less);
    if (it != kLookup + kSize && entry_eq(*it, needle))
        return it->func;
    (void)batch;
    opus_gfx1250_detail::check_shape_4g(M, N, K, sizeof(fp32_t));
    const int kid = opus_a16w16_heuristic_kid_gfx1250(M, N, K, has_bias);
    return opus_a16w16_tune_dispatch_gfx1250<fp32_t>(kid);
}
