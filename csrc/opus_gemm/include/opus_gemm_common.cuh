// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Cross-arch traits umbrella: aggregates the per-arch traits headers so the
// dispatcher TU (opus_gemm.cu) and any per-arch glue header
// (opus_gemm_arch_<arch>.cuh) get all kargs / traits types in one include.
//
// Today only gfx950 ships. When a new arch lands, add its traits headers
// here, e.g.:
//
//     #include "gfx942/opus_gemm_traits_a16w16_gfx942.cuh"
//
// The per-arch struct names (e.g. opus_gemm_noscale_kargs_gfx950) keep
// definitions from colliding when two arches' headers are visible in the
// same TU.
#pragma once

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"

#include <cstdint>
#include <initializer_list>
#include <limits>

// Host-only, family-neutral helpers for caller-owned typed workspaces.  Shape,
// architecture, kernel-id, and tile policy stay in the family launchers; this
// layer only enforces the common physical buffer contract.
inline size_t opus_checked_extent_product(std::initializer_list<size_t> extents,
                                          const char* label)
{
    size_t product = 1;
    for(const size_t extent : extents)
    {
        AITER_CHECK(extent > 0, label, ": workspace extents must be positive");
        AITER_CHECK(product <= std::numeric_limits<size_t>::max() / extent,
                    label,
                    ": workspace extent product overflows size_t");
        product *= extent;
    }
    return product;
}

inline void* opus_validate_workspace(aiter_tensor_t& workspace,
                                     const aiter_tensor_t& reference,
                                     AiterDtype expected_dtype,
                                     size_t required_numel,
                                     size_t alignment,
                                     const char* label)
{
    AITER_CHECK(required_numel > 0,
                label,
                ": required workspace element count must be positive");
    AITER_CHECK(alignment > 0 && (alignment & (alignment - 1)) == 0,
                label,
                ": workspace alignment must be a non-zero power of two");
    AITER_CHECK(workspace.device_id == reference.device_id,
                label,
                ": workspace device ",
                workspace.device_id,
                " must match input device ",
                reference.device_id);
    AITER_CHECK(workspace.dtype() == expected_dtype,
                label,
                ": workspace dtype must be ",
                AiterDtype_to_str(expected_dtype),
                ", got ",
                AiterDtype_to_str(workspace.dtype()));
    AITER_CHECK(workspace.is_contiguous(), label, ": workspace must be contiguous");
    AITER_CHECK(workspace.numel() >= required_numel,
                label,
                ": workspace capacity is ",
                workspace.numel(),
                " elements, but ",
                required_numel,
                " are required");

    void* ptr = workspace.data_ptr();
    AITER_CHECK(ptr != nullptr, label, ": workspace data pointer must be non-null");
    AITER_CHECK(reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0,
                label,
                ": workspace address must be aligned to ",
                alignment,
                " bytes");

    // The workspace is typed, so ensure its required byte span is representable
    // as well as its logical element count.
    (void)opus_checked_extent_product(
        {required_numel, workspace.element_size()}, label);
    return ptr;
}
#endif

#include "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_noscale_gfx950.cuh"
// Both opus_gemm_a16w16_traits_gfx950 (split-barrier) and
// opus_gemm_a16w16_flatmm_traits_gfx950 (warp-spec) live in this one header.
#include "gfx950/opus_gemm_traits_a16w16_gfx950.cuh"
// gfx1250 cluster/TDM split-K (workspace + reduce) traits + direct-pointer kargs.
#include "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh"
