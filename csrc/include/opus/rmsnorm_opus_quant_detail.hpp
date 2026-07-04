// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus-clean helpers for the fused rmsnorm+quant path (grouped / MXFP4 / shuffle).
// Includes ONLY opus.hpp — no torch / rocprim / hipcub / aiter_opus_plus.h — so the
// module_rmsnorm TU stays a torch-free single-TU ~1.4s build. The math below is
// inlined verbatim from mx_quant_utils.h (which is itself <cstdint>-only).
#pragma once
#include "opus/opus.hpp"

namespace aiter {

// MXFP4 E8M0 block scale, RoundUp (NV/DSv4/torchao RCEIL): ceil_pow2(amax / 6),
// with 6 = max_pos of fp4 e2m1. Matches aiter::fp4_f32_to_e8m0_scale.
__device__ __forceinline__ float fp4_e8m0_scale(float amax)
{
    const unsigned u = __builtin_bit_cast(unsigned, amax * (1.0f / 6.0f));
    unsigned e       = (u >> 23) & 0xFFu;
    if(e < 0xFFu && (u & 0x7FFFFFu))
        e += 1;
    return __builtin_bit_cast(float, e << 23);
}

// e8m0 byte = biased exponent of an fp32 power-of-two scale.
__device__ __forceinline__ unsigned char e8m0_byte(float scale)
{
    return static_cast<unsigned char>((__builtin_bit_cast(unsigned, scale) >> 23) & 0xFFu);
}

// Swizzled E8M0 scale index for the tiled MX layout (identical for MXFP4/MXFP8).
// Verbatim from aiter::mx_scale_shuffle_idx.
__device__ __forceinline__ int mx_scale_shuffle_idx(int scaleN_pad, int x, int y)
{
    return (x / 32 * scaleN_pad) * 32 + (y / 8) * 256 + (y % 4) * 64 + (x % 16) * 4 +
           (y % 8) / 4 * 2 + (x % 32) / 16;
}

} // namespace aiter
