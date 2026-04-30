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

#include "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a8w8_noscale_gfx950.cuh"
// Both opus_gemm_a16w16_traits_gfx950 (split-barrier) and
// opus_gemm_a16w16_flatmm_traits_gfx950 (warp-spec) live in this one header.
#include "gfx950/opus_gemm_traits_a16w16_gfx950.cuh"
