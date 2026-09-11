// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_arch_gfx942.cuh -- gfx942-specific dispatch implementations.
#pragma once

#include "../opus_gemm_arch.cuh"
#include "../opus_gemm_common.cuh"
#include "../opus_gemm_lookup_entry.cuh"  // OpusLookupKey + opus_lookup_find
#include "opus_gemm_heuristic_dispatch_gfx942.cuh"  // OpusA16W16NoscaleKernel + opus_a16w16_heuristic_dispatch_gfx942<>
#include "opus_gemm_lookup.h"                       // GENERATE_OPUS_LOOKUP_TABLE_{BF16,FP32}_GFX942
#include "opus_gemm_a16w16_tune_lookup.h"           // GENERATE_A16W16_TUNE_LOOKUP_{BF16,FP32}_GFX942
#include "opus_gemm_a8w8_tune_lookup.h"             // GENERATE_A8W8_TUNE_LOOKUP_BF16
#include "opus_gemm_manifest.h"                     // launcher symbols referenced by the lookup macros
#include "../opus_gemm_utils.cuh"                   // bf16_t / fp32_t

#include <algorithm>  // std::lower_bound
#include <cstddef>
#include <optional>

namespace opus_gfx942_detail
{
struct OpusA16W16RuntimeEntry
{
    OpusLookupKey key;
    OpusA16W16NoscaleKernel func;
};

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

using OpusA8W8BlockscaleBPreshuffleKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);

struct OpusA8W8TuneEntry
{
    int kid;
    OpusA8W8BlockscaleBPreshuffleKernel func;
};

constexpr bool a8w8_tune_entry_less(const OpusA8W8TuneEntry& a,
                                    const OpusA8W8TuneEntry& b) noexcept
{
    return a.kid < b.kid;
}
}  // namespace opus_gfx942_detail

// -- a16w16 runtime dispatch (tuned lookup -> heuristic fallback) -------------

template <typename CDataType>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx942(int M, int N, int K, int batch, bool has_bias = false);

template <>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx942<bf16_t>(int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx942_detail;
    static constexpr OpusA16W16RuntimeEntry kLookup[] = {
        GENERATE_OPUS_LOOKUP_TABLE_BF16_GFX942(bf16_t)
    };
    constexpr size_t kSize = sizeof(kLookup) / sizeof(kLookup[0]);
    if (auto* e = opus_lookup_find(kLookup, kLookup + kSize, M, N, K,
                                   opus_get_device_cu_num()))
    {
        return e->func;
    }
    return opus_a16w16_heuristic_dispatch_gfx942<bf16_t>(M, N, K, batch, has_bias);
}

template <>
inline OpusA16W16NoscaleKernel
opus_dispatch_a16w16_gfx942<fp32_t>(int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx942_detail;
    static constexpr OpusA16W16RuntimeEntry kLookup[] = {
        GENERATE_OPUS_LOOKUP_TABLE_FP32_GFX942(fp32_t)
    };
    constexpr size_t kSize = sizeof(kLookup) / sizeof(kLookup[0]);
    if (auto* e = opus_lookup_find(kLookup, kLookup + kSize, M, N, K,
                                   opus_get_device_cu_num()))
    {
        return e->func;
    }
    return opus_a16w16_heuristic_dispatch_gfx942<fp32_t>(M, N, K, batch, has_bias);
}

// -- a16w16 tune dispatch (id-based, two specializations) --------------------

template <typename CDataType>
inline opus_gfx942_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx942(int id);

template <>
inline opus_gfx942_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx942<bf16_t>(int id)
{
    using namespace opus_gfx942_detail;
    static constexpr OpusA16W16TuneEntry kTune[] = {
        GENERATE_A16W16_TUNE_LOOKUP_BF16_GFX942(bf16_t)
    };
    constexpr size_t kSize = sizeof(kTune) / sizeof(kTune[0]);
    OpusA16W16TuneEntry needle{id, nullptr};
    auto it = std::lower_bound(kTune, kTune + kSize, needle, tune_entry_less);
    AITER_CHECK_OR_RAISE(aiter_detail::unbaked_kernel_error,
                         it != kTune + kSize && it->kid == id,
                         "Kernel id ", id,
                         " not found in a16w16 bf16 tune lookup table (gfx942)");
    return it->func;
}

template <>
inline opus_gfx942_detail::OpusA16W16TuneKernel
opus_a16w16_tune_dispatch_gfx942<fp32_t>(int id)
{
    using namespace opus_gfx942_detail;
    static constexpr OpusA16W16TuneEntry kTune[] = {
        GENERATE_A16W16_TUNE_LOOKUP_FP32_GFX942(fp32_t)
    };
    constexpr size_t kSize = sizeof(kTune) / sizeof(kTune[0]);
    OpusA16W16TuneEntry needle{id, nullptr};
    auto it = std::lower_bound(kTune, kTune + kSize, needle, tune_entry_less);
    AITER_CHECK_OR_RAISE(aiter_detail::unbaked_kernel_error,
                         it != kTune + kSize && it->kid == id,
                         "Kernel id ", id,
                         " not found in a16w16 fp32 tune lookup table (gfx942)");
    return it->func;
}

// -- a8w8 tune dispatch (id-based, bf16-output explicit tune API only) --------

inline opus_gfx942_detail::OpusA8W8BlockscaleBPreshuffleKernel
opus_a8w8_tune_dispatch_gfx942(int id);

inline opus_gfx942_detail::OpusA8W8BlockscaleBPreshuffleKernel
opus_a8w8_tune_dispatch_gfx942(int id)
{
    using namespace opus_gfx942_detail;
    static constexpr OpusA8W8TuneEntry kTune[] = {
        GENERATE_A8W8_TUNE_LOOKUP_BF16(bf16_t)
    };
    constexpr size_t kSize = sizeof(kTune) / sizeof(kTune[0]);
    OpusA8W8TuneEntry needle{id, nullptr};
    auto it = std::lower_bound(kTune, kTune + kSize, needle, a8w8_tune_entry_less);
    AITER_CHECK_OR_RAISE(aiter_detail::unbaked_kernel_error,
                         it != kTune + kSize && it->kid == id,
                         "Kernel id ", id,
                         " not found in a8w8 bf16 tune lookup table (gfx942)");
    return it->func;
}
