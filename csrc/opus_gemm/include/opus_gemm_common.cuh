// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Umbrella header: includes all per-pipeline traits headers.
// Used by opus_gemm.cu (dispatcher) which needs all types visible.
// Individual pipeline headers should include their own traits directly.
#pragma once

#include "pipeline/opus_gemm_traits_a8w8_scale.cuh"
#include "pipeline/opus_gemm_traits_a8w8_noscale.cuh"
// Both opus_gemm_a16w16_traits (split-barrier) and
// opus_gemm_a16w16_flatmm_traits (warp-spec) live in this one header.
#include "pipeline/opus_gemm_traits_a16w16.cuh"
