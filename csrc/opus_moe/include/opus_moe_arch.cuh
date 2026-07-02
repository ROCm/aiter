// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Runtime architecture probe for Opus MoE dispatch shells. Reuses opus_gemm's
// shared OpusGfxArch probe: a new gfx target needs one per-arch dispatch header
// + one router branch, not changes to every launcher.
#pragma once

#include "opus_gemm_arch.cuh"
