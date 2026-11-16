// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <hip/hip_runtime.h>
#include <opus/opus.hpp>

namespace fused_q_norm {
// Public BF16 Q views are aligned to 16 bytes, including storage offsets.
// Keep the 16-element register fragment without requiring a 32-byte pointer.
__device__ __forceinline__ opus::vector_t<opus::bf16_t, 16>
load_bf16x16(const opus::bf16_t* ptr)
{
    opus::vector_t<opus::bf16_t, 16> result;
    opus::static_for<2>([&](auto half) {
        const auto part =
            *reinterpret_cast<const opus::vector_t<opus::bf16_t, 8>*>(ptr + int(half) * 8);
        opus::static_for<8>([&](auto i) { result[int(half) * 8 + int(i)] = part[int(i)]; });
    });
    return result;
}

// Lanes belonging to one head cooperatively reduce all 512 channels.
// The Q input remains in place; no workspace or separate kernel is needed.
template <int Lanes, int LaneStride>
__device__ __forceinline__ float
inv_rms(const opus::bf16_t* q, int fragment, bool enabled, float epsilon)
{
    if(!enabled)
        return 1.0f;
    opus::vector_t<float, 16> partial{};
    opus::static_for<512 / (Lanes * 16)>([&](auto chunk) {
        const auto x = load_bf16x16(q + int(chunk) * Lanes * 16 + fragment * 16);
        opus::static_for<16>([&](auto i) {
            const float v   = static_cast<float>(x[int(i)]);
            partial[int(i)] = fmaf(v, v, partial[int(i)]);
        });
    });
    // Pairwise sums avoid a 128-element dependency chain and reduce rounding error.
    opus::static_for<4>([&](auto step) {
        constexpr int count = 8 >> int(step);
        opus::static_for<count>([&](auto i) { partial[int(i)] += partial[int(i) + count]; });
    });
    float sum = partial[0];
    opus::static_for<__builtin_ctz(Lanes)>(
        [&](auto step) { sum += __shfl_xor(sum, LaneStride << int(step), 64); });
    return rsqrtf(sum * (1.0f / 512.0f) + epsilon);
}

template <class V>
__device__ __forceinline__ void apply(V& x, float inv)
{
    opus::static_for<sizeof(V) / sizeof(opus::bf16_t)>([&](auto i) {
        // Match a standalone BF16 RMSNorm before RoPE or Q quantization.
        x[int(i)] = static_cast<opus::bf16_t>(static_cast<float>(x[int(i)]) * inv);
    });
}
} // namespace fused_q_norm
