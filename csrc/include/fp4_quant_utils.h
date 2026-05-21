// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

namespace aiter {

// E8M0 block-scale rounding modes for the whole MX format family
// (mxfp4 / mxfp6 / mxfp8 / mxint8) -- the four formulas FLOOR / RCEIL /
// CEIL / EVEN are dtype-agnostic, only ``max_pos`` / ``max_pow2`` constants
// differ (see PyTorch torchao ``ScaleCalculationMode`` for the same design).
// Currently consumed by the mxfp4 kernel; future mx kernels can reuse this
// enum directly via ``#include "fp4_quant_utils.h"``.
//
// Names follow AMD Quark's RoundMode for AMD-side familiarity. Each value is
// 1:1 mathematically equivalent to a PyTorch torchao ScaleCalculationMode
// (see csrc/kernels/quant.md "Cross-Stack Mode Alignment Reference"):
//   Quark RoundMode (this enum) <-> torchao ScaleCalculationMode
//   RoundDown                   <-> FLOOR
//   RoundUp                     <-> RCEIL
//   Even                        <-> EVEN
//   Ceil                        <-> CEIL    (no Quark equivalent)
// Ref: Quark/quark/torch/quantization/utils.py     (RoundMode enum)
//      Quark/quark/torch/kernel/mx/triton.py        (_compute_quant_and_scale)
//      torchao/prototype/mx_formats/config.py       (ScaleCalculationMode)
//      torchao/prototype/mx_formats/mx_tensor.py    (_to_mx_rceil and friends)
enum class MxScaleRoundMode : int {
    RoundDown = 0, // OCP / NV ROUND_DOWN / torchao FLOOR:
                   //   scale = floor_pow2(amax) / 4. ~37% max clipping.
    RoundUp   = 1, // NV / DSv4 Pro / FlashInfer / torchao RCEIL:
                   //   scale = ceil_pow2(amax / 6). 0% max clipping. (industry default)
    Even      = 2, // Quark EVEN / torchao EVEN:
                   //   scale = round_pow2_1.75(amax) / 4. ~21% max clipping.
    Ceil      = 3, // torchao CEIL (no Quark / NV equivalent):
                   //   scale = ceil_pow2(amax) / 4. 0% max clipping but
                   //   coarser grid than RoundUp on [2^k, 1.5*2^k).
};

// Default MXFP4 E8M0 scale: NV ROUND_UP / DSv4 Pro / FlashInfer (industry mainstream).
// scale = ceil_pow2(amax / 6).  Guarantees 0% max-value clipping (scale * 6 >= amax).
// NaN/Inf inputs yield exponent 0xFF (E8M0 NaN); mantissa is always stripped.
__device__ __forceinline__ float fp4_f32_to_e8m0_scale(float amax)
{
    constexpr float inv_fp4_max = 1.0f / 6.0f;
    uint32_t u32      = __builtin_bit_cast(uint32_t, amax * inv_fp4_max);
    uint32_t exponent = (u32 >> 23) & 0xFFu;
    if(exponent < 0xFFu && (u32 & 0x7FFFFFu))
        exponent += 1;
    return __builtin_bit_cast(float, exponent << 23);
}

// Backward-compat alias: same as fp4_f32_to_e8m0_scale (now the default round-up form).
__device__ __forceinline__ float fp4_f32_to_e8m0_scale_roundup(float amax)
{
    return fp4_f32_to_e8m0_scale(amax);
}

// OCP standard / NV ROUND_DOWN / torchao FLOOR:
// scale = floor_pow2(amax / 4) = floor_pow2(amax) / 4.
// ~37% of random inputs will have amax > scale * 6 (max clipping).  Opt-in only;
// callers that explicitly request OCP RoundDown semantics should use this.
// NaN/Inf inputs yield exponent 0xFF (E8M0 NaN); mantissa is always stripped.
__device__ __forceinline__ float fp4_f32_to_e8m0_scale_ocp(float amax)
{
    constexpr float inv_fp4_pow2_max = 0.25f; // 1 / max_pow2(FP4 E2M1) = 1/4
    uint32_t u32      = __builtin_bit_cast(uint32_t, amax * inv_fp4_pow2_max);
    uint32_t exponent = (u32 >> 23) & 0xFFu;
    return __builtin_bit_cast(float, exponent << 23);
}

// torchao CEIL: scale = ceil_pow2(amax / 4) = ceil_pow2(amax) / 4.
// 0% max-value clipping but coarser quantization grid than the default
// round-up (which uses divisor 6).  No Quark / NV equivalent; this exists
// purely to mirror the fourth ScaleCalculationMode in PyTorch torchao.
// NaN/Inf inputs yield exponent 0xFF (E8M0 NaN); mantissa is always stripped.
__device__ __forceinline__ float fp4_f32_to_e8m0_scale_ceil(float amax)
{
    constexpr float inv_fp4_pow2_max = 0.25f; // 1 / max_pow2(FP4 E2M1) = 1/4
    uint32_t u32      = __builtin_bit_cast(uint32_t, amax * inv_fp4_pow2_max);
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
