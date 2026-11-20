// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cmath>
#include <cstdint>

namespace aiter::fmha {

inline float narrow_scale(double scale)
{
    const auto bits = __builtin_bit_cast(std::uint64_t, scale);
    if((bits & 0x7fffffffffffffffULL) >= __builtin_bit_cast(std::uint64_t, 0x1p-126))
        return static_cast<float>(scale);

    // Build the FP32 significand in units of its smallest subnormal. A direct
    // conversion here would round to zero if the host flushes subnormal results.
    const auto magnitude = static_cast<std::uint32_t>(std::fabs(std::nearbyint(scale * 0x1p149)));
    const auto sign      = static_cast<std::uint32_t>(bits >> 32) & 0x80000000u;
    return __builtin_bit_cast(float, sign | magnitude);
}

} // namespace aiter::fmha
