// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "kernel.cuh"
#include "kernel.h"

template <int Heads, bool RawQ, bool DirectQ = false, int FixedSplits = 0>
__global__ __launch_bounds__(256, 1) void fused_norm_rope_attn_rope_cast_decode_kernel(
    FusedNormRopeAttnRopeCastParams args)
{
    fused_norm_rope_attn_rope_cast_fwd::specialize_heads<Heads>(args);
    if constexpr(FixedSplits)
        args.num_splits = FixedSplits;
    using T = FusedNormRopeAttnRopeCastConfig<0, RawQ, DirectQ, FixedSplits, Heads>;
    __shared__ __align__(16) char smem[T::smem_bytes() + 2 * 68 * sizeof(int)];
    if(indices_valid<T, true>(args))
        fused_norm_rope_attn_rope_cast_fwd_impl<T, false, true, true>(args, smem);
    else
        fused_norm_rope_attn_rope_cast_fwd_impl<T, true, false, true>(args, smem);
}

// Each reduction CTA owns sixteen heads. This exposes four independent CTAs
// per attention tile and keeps only 32 output accumulators in each thread.
template <int Heads, int FixedSplits = 0>
__global__ __launch_bounds__(256, 1) void fused_norm_rope_attn_rope_cast_decode_reduce_kernel(
    FusedNormRopeAttnRopeCastParams a)
{
    using namespace opus;
    using namespace fused_norm_rope_attn_rope_cast_fwd;
    specialize_heads<Heads>(a);
    if constexpr(FixedSplits)
        a.num_splits = FixedSplits;
    const int splits = FixedSplits ? FixedSplits : a.num_splits;
    using T          = FusedNormRopeAttnRopeCastConfig<0, false>;
    const int token = blockIdx.x, h_start = blockIdx.y * 16;
    const int lane = threadIdx.x % 64, warp = threadIdx.x / 64;
    const int head         = h_start + lane % 16;
    const int head_group   = blockIdx.y % 4;
    const int group        = (head_group + 4 - warp) % 4;
    const int local_head   = head_group * 16 + lane % 16;
    constexpr float LOG2_E = 1.44269504089f, LN_2 = 0.69314718056f;
    constexpr int per_group = O_ELEMS<T> / PV_GROUPS<T>;
    vector_t<float, per_group> output{};
    const auto* base =
        a.partial_ptr +
        (int64_t(token) * a.num_splits * a.H + (Heads == 64 ? 0 : h_start / 64 * 64)) * 516;
    const int64_t split_stride = int64_t(a.H) * 516;
    float maximum = -__FLT_MAX__, true_max = -__builtin_inff();
    for(int s = 0; s < splits; ++s)
    {
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(base + s * split_stride +
                                                                        64 * 512 + local_head * 4);
        if(stats[1] > 0.0f)
            maximum = fmaxf(maximum, stats[0]);
        true_max = fmaxf(true_max, stats[2]);
    }
    // Add the sink once, after combining partials. Max/LSE exclude the sink.
    const float sink = a.attn_sink_ptr ? static_cast<const float*>(a.attn_sink_ptr)[head] * LOG2_E
                                       : -__builtin_inff();
    const float final_max = fmaxf(maximum, sink);
    float denominator = 0.0f, attention_sum = 0.0f;
    for(int s = 0; s < splits; ++s)
    {
        const auto stats   = *reinterpret_cast<const vector_t<float, 4>*>(base + s * split_stride +
                                                                        64 * 512 + local_head * 4);
        const float factor = stats[1] > 0.0f ? __builtin_amdgcn_exp2f(stats[0] - final_max) : 0.0f;
        denominator += stats[1] * factor;
        if(stats[1] > 0.0f)
            attention_sum += stats[1] * __builtin_amdgcn_exp2f(stats[0] - maximum);
        static_for<per_group / 4>([&](auto i) {
            const int chunk   = group * (per_group / 4) + i.value;
            const auto values = *reinterpret_cast<const vector_t<float, 4>*>(
                base + s * split_stride + chunk * 256 * 4 + threadIdx.x * 4);
            static_for<4>(
                [&](auto j) { output[i.value * 4 + j.value] += values[j.value] * factor; });
        });
    }
    if(a.attn_sink_ptr)
        denominator += __builtin_amdgcn_exp2f(sink - final_max);
    const float reciprocal = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    output *= reciprocal;
    if(lane < 16 && warp == 0)
    {
        if(a.max_ptr)
            a.max_ptr[int64_t(token) * a.H + head] = true_max * LN_2;
        if(a.lse_ptr)
            static_cast<float*>(a.lse_ptr)[int64_t(token) * a.H + head] =
                attention_sum > 0.0f ? (maximum + __builtin_amdgcn_logf(attention_sum)) * LN_2
                                     : __builtin_inff();
    }
    fused_epilogue<T, decltype(output), true>(a, output, warp, lane, token, h_start);
}
