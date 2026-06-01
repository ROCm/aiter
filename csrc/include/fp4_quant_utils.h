// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

namespace aiter {

// Round-to-power-of-2 with 1.5x threshold for E8M0 scale derivation.
// Matches fp4_utils.f32_to_e8m0 in Python.
//
// Note: name is dtype-neutral on purpose. The same primitive is used for
// both MXFP4 (max_pos = 6) and MXFP8 (max_pos = 448) callers -- only the
// `inverted_DTYPE_MAX` they multiply by differs. The legacy
// `fp4_f32_to_e8m0_scale` symbol is retained below as an alias for
// backward compatibility.
//
// If the float mantissa >= 0.5 (i.e. value >= 1.5 x 2^k for some k),
// the exponent is bumped by 1, rounding up to the next power of 2.
// Otherwise, the value is truncated down (floor) to a power of 2.
//
// Special: Inf/NaN (exponent == 0xFF) passes through unchanged.
//          Denormal with mantissa == exactly 0.5 is not bumped (guarded by exponent > 0).
__device__ __forceinline__ float f32_to_e8m0_scale(float x)
{
    uint32_t u32      = __builtin_bit_cast(uint32_t, x);
    uint32_t exponent = (u32 >> 23) & 0xFFu;
    if(exponent == 0xFFu)
        return __builtin_bit_cast(float, exponent << 23);
    if((u32 & 0x400000u) && ((u32 & 0x200000u) || (u32 & 0x1FFFFFu) || exponent))
        exponent += 1;
    return __builtin_bit_cast(float, exponent << 23);
}

// Backward-compat alias. New code should call `f32_to_e8m0_scale` directly.
__device__ __forceinline__ float fp4_f32_to_e8m0_scale(float x)
{
    return f32_to_e8m0_scale(x);
}

// Compute the swizzled E8M0 scale index for the tiled MX layout.
// Used by both MXFP4 and MXFP8 paths (the e8m0 byte layout is identical
// regardless of the element dtype). The legacy `fp4_scale_shuffle_idx`
// symbol is kept as an alias below.
__device__ __forceinline__ int mx_scale_shuffle_idx(int scaleN_pad, int x, int y)
{
    return (x / 32 * scaleN_pad) * 32 + (y / 8) * 256 + (y % 4) * 64 + (x % 16) * 4 +
           (y % 8) / 4 * 2 + (x % 32) / 16;
}

// Backward-compat alias. New code should call `mx_scale_shuffle_idx` directly.
__device__ __forceinline__ int fp4_scale_shuffle_idx(int scaleN_pad, int x, int y)
{
    return mx_scale_shuffle_idx(scaleN_pad, x, y);
}

} // namespace aiter
