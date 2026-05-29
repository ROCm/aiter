// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

namespace aiter {

// E8M0 block-scale rounding modes for the whole MX format family
// (mxfp4 / mxfp6 / mxfp8 / mxint8) -- the four formulas FLOOR / RCEIL /
// CEIL / EVEN are dtype-agnostic, only ``max_pos`` / ``max_pow2`` constants
// differ (see PyTorch torchao ``ScaleCalculationMode`` for the same design).
//
// Names follow AMD Quark's RoundMode for AMD-side familiarity. Each value is
// 1:1 mathematically equivalent to a PyTorch torchao ScaleCalculationMode
// (cross-stack mapping):
//   Quark RoundMode (this enum) <-> torchao ScaleCalculationMode
//   RoundDown                   <-> FLOOR
//   RoundUp                     <-> RCEIL
//   Even                        <-> EVEN
//   Ceil                        <-> CEIL    (no Quark equivalent)
// Ref: Quark/quark/torch/quantization/utils.py     (RoundMode enum)
//      Quark/quark/torch/kernel/mx/triton.py        (_compute_quant_and_scale)
//      torchao/prototype/mx_formats/config.py       (ScaleCalculationMode)
//      torchao/prototype/mx_formats/mx_tensor.py    (_to_mx_rceil and friends)
//
// Values **must** stay 1:1 with ``aiter.utility.mx_types.MxScaleRoundMode``
// (the pybind11 binding) and ``aiter.utility.mx_types.MxScaleRoundModeInt``
// (the JIT-free int mirror used by FlyDSL AOT). The lazy loader in
// ``mx_types.py::__getattr__`` asserts the int values match on first
// pybind enum access.
enum class MxScaleRoundMode : int {
    RoundDown = 0, // OCP / NV ROUND_DOWN / torchao FLOOR:
                   //   scale = floor_pow2(amax) / 2^target_max_pow2.
                   //   ~37% max clipping (FP4 default).
    RoundUp   = 1, // NV / DSv4 Pro / FlashInfer / torchao RCEIL:
                   //   scale = ceil_pow2(amax / max_pos).
                   //   0% max clipping. (industry default)
    Even      = 2, // Quark EVEN / torchao EVEN:
                   //   scale = floor_pow2(round_pow2_special(amax)) /
                   //   2^target_max_pow2 with val_to_add =
                   //   1 << (23 - mbits - 1).
                   //   AMD Quark default for MXFP4/MXFP8.
    Ceil      = 3, // torchao CEIL (no Quark / NV equivalent):
                   //   scale = ceil_pow2(amax) / 2^target_max_pow2.
                   //   0% max clipping but coarser grid than RoundUp on
                   //   [2^k, 1.5*2^k).
};

constexpr MxScaleRoundMode kDefaultMxScaleRoundMode = MxScaleRoundMode::RoundUp;

// MX-format element dtype tag selecting per-dtype constants.
// Values **must** stay 1:1 with ``aiter.utility.mx_types.MxDtype`` (the
// pybind11 binding in ``rocm_ops.hpp::AITER_CORE_PYBIND``).
//
// Note that ``FP8_E4M3`` (a.k.a. ``e4m3fn``) and ``FP8_E4M3_FNUZ`` are
// **different formats** with different ``max_pos`` (448 vs 240) and
// different ``target_max_pow2`` (8 vs 7); they are NOT interchangeable.
//   * ``FP8_E4M3``      = OCP / NVIDIA H100 / AMD gfx950+ / FlashInfer.
//                         exp_bias = 7, max_normal = 448.
//   * ``FP8_E4M3_FNUZ`` = AMD gfx942 hardware FP8 (MI300 family).
//                         exp_bias = 8, max_normal = 240.
enum class MxDtype : int {
    FP4_E2M1      = 0, // max_pos = 6.0,    target_max_pow2 = 2,  mbits = 1
    FP8_E4M3      = 1, // max_pos = 448.0,  target_max_pow2 = 8,  mbits = 3
    FP8_E4M3_FNUZ = 2, // max_pos = 240.0,  target_max_pow2 = 7,  mbits = 3
    // FP8_E5M2 / FP6_* / MX_INT8 -- reserved.
};

// Per-MX-dtype constants. The four scale rounding formulas depend on:
//   * max_pos        : max representable normal value of the target dtype
//                      (used by RoundUp / RCEIL: ceil_pow2(amax / max_pos)).
//   * target_max_pow2: log2(largest pow2 <= max_pos)
//                      (used by RoundDown / Ceil / Even: divisor
//                      2^target_max_pow2).
//   * mbits          : mantissa bits of the target dtype
//                      (used by Even: val_to_add = 1 << (23 - mbits - 1)).
template <MxDtype dtype> struct MxDtypeConfig;

template <> struct MxDtypeConfig<MxDtype::FP4_E2M1> {
    static constexpr int      target_max_pow2 = 2;        // log2(4)
    static constexpr float    max_pos         = 6.0f;
    static constexpr float    inv_max_pos     = 1.0f / 6.0f;   // 0x3E2AAAAB
    static constexpr float    inv_max_pow2    = 0.25f;         // 1/4
    static constexpr int      mbits           = 1;
    static constexpr uint32_t even_val_to_add = 0x00200000u;   // 1 << (23 - 1 - 1) = 1 << 21
};

template <> struct MxDtypeConfig<MxDtype::FP8_E4M3> {
    static constexpr int      target_max_pow2 = 8;              // log2(256), 256 <= 448
    static constexpr float    max_pos         = 448.0f;
    static constexpr float    inv_max_pos     = 1.0f / 448.0f;  // 0x3B124925
    static constexpr float    inv_max_pow2    = 1.0f / 256.0f;  // 1/256 = 0x3B800000
    static constexpr int      mbits           = 3;
    static constexpr uint32_t even_val_to_add = 0x00080000u;    // 1 << (23 - 3 - 1) = 1 << 19
};

// AMD gfx942 hardware FP8 (e4m3fnuz): exp_bias = 8 (one bigger than the
// OCP e4m3fn = 7), so max_normal = 240.0 (= 1.875 * 2^7) and the
// largest pow-2 fitting under it is 128 = 2^7. Mantissa width is the
// same 3 bits, so EVEN's val_to_add is identical to FP8_E4M3.
template <> struct MxDtypeConfig<MxDtype::FP8_E4M3_FNUZ> {
    static constexpr int      target_max_pow2 = 7;              // log2(128), 128 <= 240
    static constexpr float    max_pos         = 240.0f;
    static constexpr float    inv_max_pos     = 1.0f / 240.0f;  // 0x3B888889
    static constexpr float    inv_max_pow2    = 1.0f / 128.0f;  // 1/128 = 0x3C000000
    static constexpr int      mbits           = 3;
    static constexpr uint32_t even_val_to_add = 0x00080000u;    // 1 << (23 - 3 - 1) = 1 << 19
};

// Generic E8M0 dequant scale computation.
//
// Returns the *dequantization* scale as an f32 with a power-of-2 bit
// pattern, ready for direct multiplication into the dequantized data and
// for E8M0 bit extraction via ``(__float_as_uint(s) >> 23) & 0xFF``.
//
// This template mirrors PyTorch torchao's ``to_mx(scaling_mode, elem_dtype)``
// 1:1 and is the C++ analogue of:
//   * Python CPU ref: ``aiter.utility.fp4_utils.f32_to_mx_e8m0_scale``
//   * FlyDSL builder: ``aiter.ops.flydsl.kernels.quant_utils.emit_mx_e8m0_scale``
//     (uses the JIT-free ``MxScaleRoundModeInt`` / ``MxDtypeInt`` mirrors so
//     wheel ``PREBUILD_KERNELS`` can AOT-compile FlyDSL kernels before HIP)
//
// The template parameters are compile-time constants (selected by the
// caller's switch over runtime ``round_mode``), so the ``if constexpr``
// branch is fully resolved before codegen -- no runtime dispatch overhead.
//
// NaN/Inf inputs preserve the f32 exponent (0xFF) which downstream consumers
// interpret as E8M0 NaN.
template <MxScaleRoundMode rmode = kDefaultMxScaleRoundMode,
          MxDtype          dtype = MxDtype::FP4_E2M1>
__device__ __forceinline__ float fp_f32_to_e8m0_scale(float amax)
{
    using Cfg = MxDtypeConfig<dtype>;

    if constexpr (rmode == MxScaleRoundMode::RoundUp)
    {
        // ceil_pow2(amax / max_pos): NV / DSv4 / FlashInfer / torchao RCEIL.
        const uint32_t u32      = __builtin_bit_cast(uint32_t, amax * Cfg::inv_max_pos);
        uint32_t       exponent = (u32 >> 23) & 0xFFu;
        if(exponent < 0xFFu && (u32 & 0x7FFFFFu))
            exponent += 1;
        return __builtin_bit_cast(float, exponent << 23);
    }
    else if constexpr (rmode == MxScaleRoundMode::RoundDown)
    {
        // floor_pow2(amax) / 2^target_max_pow2: OCP MX / torchao FLOOR.
        const uint32_t u32      = __builtin_bit_cast(uint32_t, amax * Cfg::inv_max_pow2);
        const uint32_t exponent = (u32 >> 23) & 0xFFu;
        return __builtin_bit_cast(float, exponent << 23);
    }
    else if constexpr (rmode == MxScaleRoundMode::Ceil)
    {
        // ceil_pow2(amax) / 2^target_max_pow2: torchao CEIL.
        const uint32_t u32      = __builtin_bit_cast(uint32_t, amax * Cfg::inv_max_pow2);
        uint32_t       exponent = (u32 >> 23) & 0xFFu;
        if(exponent < 0xFFu && (u32 & 0x7FFFFFu))
            exponent += 1;
        return __builtin_bit_cast(float, exponent << 23);
    }
    else if constexpr (rmode == MxScaleRoundMode::Even)
    {
        // floor_pow2(round_pow2_special(amax)) / 2^target_max_pow2:
        // Quark EVEN / torchao EVEN. Add a half-step at the
        // ``(mbits+1)``-th-from-top mantissa bit, drop mantissa, then
        // subtract target_max_pow2 from the biased exponent (saturating
        // to >= 0). NaN/Inf input preserves exponent 0xFF.
        const uint32_t bits     = __builtin_bit_cast(uint32_t, amax);
        const uint32_t rounded  = (bits + Cfg::even_val_to_add) & 0xFF800000u;
        const uint32_t raw_exp  = (rounded >> 23) & 0xFFu;
        const uint32_t exponent = (raw_exp >= static_cast<uint32_t>(Cfg::target_max_pow2))
                                      ? (raw_exp - static_cast<uint32_t>(Cfg::target_max_pow2))
                                      : 0u;
        return __builtin_bit_cast(float, exponent << 23);
    }
}

// Default MXFP4 E8M0 scale helper: NV ROUND_UP / DSv4 / FlashInfer / torchao
// RCEIL with FP4 E2M1 constants, i.e. ``ceil_pow2(amax / 6)``. This is the
// industry-default MXFP4 block-scale formula and is preserved as a named
// alias for readability / 1:1 mapping with the Python helper
// :func:`aiter.utility.fp4_utils.fp4_f32_to_e8m0_scale`.
__device__ __forceinline__ float fp4_f32_to_e8m0_scale(float amax)
{
    return fp_f32_to_e8m0_scale<kDefaultMxScaleRoundMode, MxDtype::FP4_E2M1>(amax);
}

// Compute the swizzled E8M0 scale index for the tiled MX layout.
// Used by both MXFP4 and MXFP8 paths (the e8m0 byte layout is identical
// regardless of the element dtype). Legacy `fp4_scale_shuffle_idx` kept as
// an alias below.
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
