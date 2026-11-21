// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "fmha_fwd_softmax_scale.hpp"
#include <bit>
#include <opus/opus.hpp>

namespace fmha_bf16_opus {

template <typename T, typename V>
__device__ inline float tile_max(const V& scores)
{
    static_assert(T::W_M == 32);
    // A finite seed keeps masked scores at -inf after centering an empty tile.
    float row_max = opus::numeric_limits<float>::lowest();
    opus::static_for<opus::vector_traits<V>::size()>(
        [&](auto i) { row_max = opus::max(row_max, scores[i.value]); });
    // The other half of this row is 32 lanes away.
    const auto bits = std::bit_cast<opus::u32_t>(row_max);
    const auto pair = __builtin_amdgcn_permlane32_swap(bits, bits, false, true);
    return opus::max(std::bit_cast<float>(pair.x), std::bit_cast<float>(pair.y));
}

template <bool Wide, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline void center_scores(V& scores, float row_max, float scale)
{
    opus::static_for<Count>([&](auto i) {
        constexpr opus::index_t idx = Offset + i.value;
        scores[idx] = aiter::fmha::scaled_difference<Wide>(scores[idx], row_max, scale);
        asm volatile("" : "+v"(scores[idx]));
    });
}

} // namespace fmha_bf16_opus
