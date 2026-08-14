// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Exact-kid launcher tables for gfx1250.
#pragma once

#include "../opus_gemm_arch.cuh"
#include "../opus_gemm_common.cuh"
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_a16w16_kid_dispatch.h"
#include "opus_gemm_a8w8_kid_dispatch.h"
#include "opus_gemm_manifest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

#ifndef OPUS_A16W16_DISPATCH_KERNEL_TYPES_DEFINED
#define OPUS_A16W16_DISPATCH_KERNEL_TYPES_DEFINED
using OpusA16W16Kernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    std::optional<aiter_tensor_t>, int);
using OpusA16W16WorkspaceKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    aiter_tensor_t&, std::optional<aiter_tensor_t>, int);
#endif

#ifndef OPUS_A8W8_DISPATCH_KERNEL_TYPES_DEFINED
#define OPUS_A8W8_DISPATCH_KERNEL_TYPES_DEFINED
using OpusA8W8Kernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&);
using OpusA8W8BlockscaleKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    aiter_tensor_t&, aiter_tensor_t&);
using OpusA8W8BlockscaleBpreshuffleKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    aiter_tensor_t&, aiter_tensor_t&);
#endif

namespace opus_gfx1250_detail
{
struct OpusA16W16KidEntry
{
    int kid;
    OpusA16W16Kernel func;
};

struct OpusA16W16WorkspaceKidEntry
{
    int kid;
    OpusA16W16WorkspaceKernel func;
};

template <typename Kernel>
struct OpusA8W8KidEntry
{
    int kid;
    Kernel func;
};

template <typename Entry, size_t Size>
inline const Entry* find_kid(const std::array<Entry, Size>& entries, int kid)
{
    const auto it = std::lower_bound(
        entries.begin(), entries.end(), kid,
        [](const Entry& entry, int value) { return entry.kid < value; });
    return it != entries.end() && it->kid == kid ? &*it : nullptr;
}

inline const OpusA16W16WorkspaceKidEntry* workspace_entry(int kid)
{
    static constexpr std::array<
        OpusA16W16WorkspaceKidEntry,
        GENERATE_A16W16_WORKSPACE_KID_DISPATCH_GFX1250_SIZE>
        kWorkspace = {{GENERATE_A16W16_WORKSPACE_KID_DISPATCH_GFX1250}};
    return find_kid(kWorkspace, kid);
}
} // namespace opus_gfx1250_detail

template <typename CDataType>
inline OpusA8W8BlockscaleBpreshuffleKernel
opus_a8w8_blockscale_bpreshuffle_kid_dispatch_gfx1250(int id);

template <>
inline OpusA8W8BlockscaleBpreshuffleKernel
opus_a8w8_blockscale_bpreshuffle_kid_dispatch_gfx1250<bf16_t>(int id)
{
    using Entry = opus_gfx1250_detail::OpusA8W8KidEntry<
        OpusA8W8BlockscaleBpreshuffleKernel>;
    static constexpr std::array<
        Entry,
        GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_BF16_SIZE>
        kKids = {{GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_BF16}};
    AITER_CHECK(!kKids.empty(),
                "no registered kernel for OPUS "
                "a8w8_blockscale_bpreshuffle on gfx1250 with bf16 Y");
    const auto* entry = opus_gfx1250_detail::find_kid(kKids, id);
    AITER_CHECK(entry != nullptr,
                "unknown kid ", id, " for OPUS "
                "a8w8_blockscale_bpreshuffle on gfx1250 with bf16 Y");
    return entry->func;
}

template <>
inline OpusA8W8BlockscaleBpreshuffleKernel
opus_a8w8_blockscale_bpreshuffle_kid_dispatch_gfx1250<fp32_t>(int id)
{
    using Entry = opus_gfx1250_detail::OpusA8W8KidEntry<
        OpusA8W8BlockscaleBpreshuffleKernel>;
    static constexpr std::array<
        Entry,
        GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_FP32_SIZE>
        kKids = {{GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX1250_FP32}};
    AITER_CHECK(!kKids.empty(),
                "no registered kernel for OPUS "
                "a8w8_blockscale_bpreshuffle on gfx1250 with fp32 Y");
    const auto* entry = opus_gfx1250_detail::find_kid(kKids, id);
    AITER_CHECK(entry != nullptr,
                "unknown kid ", id, " for OPUS "
                "a8w8_blockscale_bpreshuffle on gfx1250 with fp32 Y");
    return entry->func;
}

// Empty direct-output tables reject unsupported gfx1250 A16W16 calls.
template <typename CDataType>
inline OpusA16W16Kernel opus_a16w16_kid_dispatch_gfx1250(int kid);

template <>
inline OpusA16W16Kernel opus_a16w16_kid_dispatch_gfx1250<bf16_t>(int kid)
{
    using namespace opus_gfx1250_detail;
    static constexpr std::array<
        OpusA16W16KidEntry,
        GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX1250_BF16_SIZE>
        kKids = {{GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX1250_BF16(bf16_t)}};
    const auto* entry = find_kid(kKids, kid);
    AITER_CHECK(entry != nullptr,
                "unknown kid ", kid,
                " for OPUS a16w16 on gfx1250 with bf16 Y in the "
                "non-workspace launch table");
    return entry->func;
}

template <>
inline OpusA16W16Kernel opus_a16w16_kid_dispatch_gfx1250<fp32_t>(int kid)
{
    using namespace opus_gfx1250_detail;
    static constexpr std::array<
        OpusA16W16KidEntry,
        GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX1250_FP32_SIZE>
        kKids = {{GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_GFX1250_FP32(fp32_t)}};
    const auto* entry = find_kid(kKids, kid);
    AITER_CHECK(entry != nullptr,
                "unknown kid ", kid,
                " for OPUS a16w16 on gfx1250 with fp32 Y in the "
                "non-workspace launch table");
    return entry->func;
}

inline bool opus_a16w16_has_workspace_kernel_gfx1250(int id)
{
    return opus_gfx1250_detail::workspace_entry(id) != nullptr;
}

inline OpusA16W16WorkspaceKernel
opus_a16w16_workspace_dispatch_gfx1250(int id)
{
    const auto* entry = opus_gfx1250_detail::workspace_entry(id);
    AITER_CHECK(entry != nullptr,
                "unknown kid ", id,
                " for OPUS a16w16 on gfx1250 in the workspace launch table");
    return entry->func;
}
