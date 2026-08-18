// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Runtime architecture probe for Opus MoE dispatch shells.
//
// Keep this layer aligned with opus_gemm: opus_moe reuses the shared OpusGfxArch
// probe so adding a new gfx target means adding one per-arch dispatch header and
// one router branch, not changing every launcher.
#pragma once

#include "opus_gemm_arch.cuh"
