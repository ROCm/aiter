// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../opus_moe_arch_gfx950.cuh"
#include "opus_moe_stage2_a8w4_manifest.h"

inline void opus_moe_stage2_a8w4_decode_dispatch_gfx950(
    int kid,
    int effective_inter_dim,
    const opus_moe_stage2_a8w4_kargs& kargs,
    hipStream_t stream)
{
    switch(kid)
    {
    GENERATE_OPUS_MOE_STAGE2_A8W4_DECODE_DISPATCH_CASES
    default: break;
    }
    AITER_CHECK(false,
                "unreachable A8W4 kernel dispatch for kid=",
                kid,
                " effective_inter_dim=",
                effective_inter_dim);
}
