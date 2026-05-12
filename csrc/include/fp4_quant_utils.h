// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

namespace aiter {

// Round-to-power-of-2 with 1.5x threshold for E8M0 scale derivation.
// Matches fp4_utils.f32_to_e8m0 in Python.
//
// If the float mantissa >= 0.5 (i.e. value >= 1.5 x 2^k for some k),
// the exponent is bumped by 1, rounding up to the next power of 2.
// Otherwise, the value is truncated down (floor) to a power of 2.
//
// Special: Inf/NaN (exponent == 0xFF) passes through unchanged.
//          Denormal with mantissa == exactly 0.5 is not bumped (guarded by exponent > 0).
__device__ __forceinline__ float fp4_round_pow2(float x)
{
    uint32_t u32      = __builtin_bit_cast(uint32_t, x);
    uint32_t exponent = (u32 >> 23) & 0xFFu;
    if(exponent == 0xFFu)
        return __builtin_bit_cast(float, exponent << 23);
    if((u32 & 0x400000u) && ((u32 & 0x200000u) || (u32 & 0x1FFFFFu) || exponent))
        exponent += 1;
    return __builtin_bit_cast(float, exponent << 23);
}

// Compute the swizzled E8M0 scale index for tiled FP4 layout.
// Used when shuffle_scale is enabled for MXFP4 quantization.
__device__ __forceinline__ int fp4_scale_shuffle_idx(int scaleN_pad, int x, int y)
{
    return (x / 32 * scaleN_pad) * 32 + (y / 8) * 256 + (y % 4) * 64 + (x % 16) * 4 +
           (y % 8) / 4 * 2 + (x % 32) / 16;
}

} // namespace aiter
