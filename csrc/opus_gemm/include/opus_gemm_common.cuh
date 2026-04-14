// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Umbrella header: includes all per-pipeline traits headers.
// Used by opus_gemm.cu (dispatcher) which needs all types visible.
// Individual pipeline headers should include their own traits directly.
#pragma once

#include "pipeline/opus_gemm_traits_a8w8_scale.cuh"
#include "pipeline/opus_gemm_traits_a8w8_noscale.cuh"
#include "pipeline/opus_gemm_traits_a16w16.cuh"
