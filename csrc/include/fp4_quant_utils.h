// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

namespace aiter {

// OCP standard: E8M0 scale = floor_pow2(amax / 4) = floor_pow2(amax) / 4.
// ~37% of random inputs will have amax > scale * 6 (max clipping).
// NaN/Inf inputs yield exponent 0xFF (E8M0 NaN); mantissa is always stripped.
__device__ __forceinline__ float fp4_f32_to_e8m0_scale(float amax)
{
    constexpr float inv_fp4_pow2_max = 0.25f; // 1 / max_pow2(FP4 E2M1) = 1/4
    uint32_t u32      = __builtin_bit_cast(uint32_t, amax * inv_fp4_pow2_max);
    uint32_t exponent = (u32 >> 23) & 0xFFu;
    return __builtin_bit_cast(float, exponent << 23);
}

// NV ROUND_UP / DSv4 Pro / FlashInfer: E8M0 scale = ceil_pow2(amax / 6).
// Guarantees 0% max-value clipping: scale * 6 >= amax always holds.
// NaN/Inf inputs yield exponent 0xFF (E8M0 NaN); mantissa is always stripped.
__device__ __forceinline__ float fp4_f32_to_e8m0_scale_roundup(float amax)
{
    constexpr float inv_fp4_max = 1.0f / 6.0f;
    uint32_t u32      = __builtin_bit_cast(uint32_t, amax * inv_fp4_max);
    uint32_t exponent = (u32 >> 23) & 0xFFu;
    if(exponent < 0xFFu && (u32 & 0x7FFFFFu))
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
