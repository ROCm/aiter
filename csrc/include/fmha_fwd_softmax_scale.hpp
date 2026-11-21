// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

namespace aiter::fmha {

// A raw FP32 difference can overflow. Above 2^-120, that gap already
// saturates exp2. Below 2^64, flushing a subnormal gap changes a weight by
// less than 2^-62. Select the wider arithmetic once at launch.
__host__ __device__ inline bool requires_wide_scale(float scale)
{
    const float magnitude = __builtin_fabsf(scale);
    return magnitude < 0x1p-120f || magnitude >= 0x1p64f;
}

// Widen a nonnegative scale without flushing a subnormal input to zero.
__device__ __forceinline__ double wide_scale(float scale)
{
    unsigned bits = __builtin_bit_cast(unsigned, scale);
    asm("" : "+v"(bits));
    return bits < 0x00800000u ? static_cast<double>(bits) * 0x1p-149 : static_cast<double>(scale);
}

template <bool Wide>
__device__ __forceinline__ float scaled_difference(float score, float row_max, float scale)
{
    // Center first so nearby scores keep their difference under fast-math.
    if constexpr(Wide)
    {
        double delta = static_cast<double>(score) - static_cast<double>(row_max);
        asm("" : "+v"(delta));
        return static_cast<float>(delta * (wide_scale(scale) * 1.4426950408889634));
    }
    else
    {
        float delta = score - row_max;
        asm("" : "+v"(delta));
        return delta * (scale * 1.4426950408889634f);
    }
}

template <bool Wide>
__device__ __forceinline__ float softmax_lse(float row_max, float scale, float log2_sum)
{
    const float log_sum = log2_sum * 0.6931471805599453f;
    if constexpr(Wide)
        return static_cast<float>(wide_scale(scale) * static_cast<double>(row_max) + log_sum);
    return __builtin_fmaf(row_max, scale, log_sum);
}

} // namespace aiter::fmha
