// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Split-output reduction and launchers shared by the 64-row attention kernels.
#include "attention_64rows.cuh"

template <int Heads, int FixedSplits = 0, bool OutputOnly = false, int HeadTile = 64>
__global__ __launch_bounds__(256, 1) void fused_norm_rope_attn_rope_cast_decode_reduce_kernel(
    FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    using namespace fused_norm_rope_attn_rope_cast_fwd;
    specialize_heads<Heads>(a);
    if constexpr(FixedSplits)
        a.num_splits = FixedSplits;
    if constexpr(OutputOnly)
    {
        a.debug_ptr = nullptr;
        a.lse_ptr   = nullptr;
        a.max_ptr   = nullptr;
        a.out_ptr   = nullptr;
    }
    const int splits             = FixedSplits ? FixedSplits : a.num_splits;
    using T                      = FusedNormRopeAttnRopeCastConfig<false, 0, Heads, HeadTile>;
    constexpr int partial_heads  = Heads < HeadTile ? Heads : HeadTile;
    constexpr int partial_groups = partial_heads / 16;
    const int token = blockIdx.x, h_start = blockIdx.y * 16;
    const int lane = threadIdx.x % 64, warp = threadIdx.x / 64;
    const int head         = h_start + lane % 16;
    const int head_group   = blockIdx.y % partial_groups;
    const int group        = (head_group + partial_groups - warp % partial_groups) % partial_groups;
    const int local_head   = head_group * 16 + lane % 16;
    constexpr float LOG2_E = 1.44269504089f, LN_2 = 0.69314718056f;
    constexpr int per_group = O_ELEMS<T> / PV_GROUPS<T>;
    vector_t<float, per_group> output{};
    const auto* base = a.partial_ptr + (int64_t(token) * a.num_splits * a.H +
                                        (Heads <= HeadTile ? 0 : h_start / HeadTile * HeadTile)) *
                                           516;
    const int64_t split_stride = int64_t(a.H) * 516;
    float maximum = -__FLT_MAX__, true_max = -__builtin_inff();
    for(int s = 0; s < splits; ++s)
    {
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(
            base + s * split_stride + partial_heads * 512 + local_head * 4);
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
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(
            base + s * split_stride + partial_heads * 512 + local_head * 4);
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
#endif
}

template <int Splits>
__global__ __launch_bounds__(128) void fused_norm_rope_attn_rope_cast_reduce_head_kernel(
    FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    constexpr int Values = 4;
    const int head = blockIdx.y, token = blockIdx.x;
    const int lane = threadIdx.x % 64, d = threadIdx.x * Values;
    const int local_head = head % 64;
    const int64_t stride = int64_t(a.H) * 516;
    const float* base = a.partial_ptr + int64_t(token) * Splits * stride + (head / 64) * 64 * 516;
    const auto stats  = *reinterpret_cast<const vector_t<float, 4>*>(
        base + (lane < Splits ? lane : 0) * stride + 64 * 512 + local_head * 4);
    const float sum = lane < Splits ? stats[1] : 0.0f;
    float maximum   = sum > 0.0f ? stats[0] : -__FLT_MAX__;
    for(int offset = 32; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset));
    const float sink   = a.attn_sink_ptr
                             ? static_cast<const float*>(a.attn_sink_ptr)[head] * 1.44269504089f
                             : -__builtin_inff();
    maximum            = fmaxf(maximum, sink);
    const float factor = sum > 0.0f ? __builtin_amdgcn_exp2f(stats[0] - maximum) : 0.0f;
    float denominator  = sum * factor;
    for(int offset = 32; offset > 0; offset /= 2)
        denominator += __shfl_xor(denominator, offset);
    if(a.attn_sink_ptr)
        denominator += __builtin_amdgcn_exp2f(sink - maximum);
    const float reciprocal = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    vector_t<float, Values> output{};
    static_for<Splits>([&](auto s) {
        const float w = __shfl(factor, s.value);
        static_for<Values / 4>([&](auto v) {
            const int col         = d + v.value * 4;
            const int warp        = col / 128;
            const int source_lane = local_head % 16 + (col % 16) / 4 * 16;
            const int group       = (local_head / 16 + 4 - warp) % 4;
            const int chunk       = group * 8 + (col % 128) / 16;
            const auto x          = *reinterpret_cast<const vector_t<float, 4>*>(
                base + s.value * stride + chunk * 1024 + (warp * 64 + source_lane) * 4);
            static_for<4>([&](auto j) {
                output[v.value * 4 + j.value] =
                    __builtin_fmaf(x[j.value], w, output[v.value * 4 + j.value]);
            });
        });
    });
    output = cast<float>(cast<dtypes::bf16>(output * reciprocal));
    if(d >= 448)
    {
        const int64_t position = a.positions64 ? static_cast<const int64_t*>(a.positions_ptr)[token]
                                               : static_cast<const int*>(a.positions_ptr)[token];
        const auto* cosine =
            static_cast<const dtypes::bf16*>(a.cos_ptr) + position * 32 + (d - 448) / 2;
        const auto* sine =
            static_cast<const dtypes::bf16*>(a.sin_ptr) + position * 32 + (d - 448) / 2;
        static_for<Values / 2>([&](auto i) {
            constexpr int j = i.value * 2;
            const float c = cast<float>(cosine[i.value]), s = cast<float>(sine[i.value]);
            const float even = output[j], odd = output[j + 1];
            output[j]     = even * c + odd * s;
            output[j + 1] = odd * c - even * s;
        });
    }
    float maximum_abs = 1e-8f;
    static_for<Values>([&](auto i) { maximum_abs = fmaxf(maximum_abs, fabsf(output[i.value])); });
    for(int offset = 1; offset < 32 / Values; offset *= 2)
        maximum_abs = fmaxf(maximum_abs, __shfl_xor(maximum_abs, offset));
    const unsigned bits    = std::bit_cast<unsigned>(maximum_abs / 448.0f);
    const int exponent     = int((bits - 1) >> 23) + 1 - 127;
    const float multiplier = std::bit_cast<float>(unsigned(127 - exponent) << 23);
    const auto quantized   = cast<opus::fp8_t>(output * multiplier);
    const int64_t row      = int64_t(token) * a.H + head;
    *reinterpret_cast<vector_t<opus::fp8_t, Values>*>(a.output_q_ptr + row * 512 + d) = quantized;
    if(d % 32 == 0)
        a.output_s_ptr[row * 16 + d / 32] = exponent + 127;
#endif
}

template <int Splits>
__global__ __launch_bounds__(64) void fused_norm_rope_attn_rope_cast_reduce_parallel_kernel(
    FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    using namespace fused_norm_rope_attn_rope_cast_fwd;
    using T               = FusedNormRopeAttnRopeCastConfig<>;
    constexpr int Entries = (Splits + 3) / 4;
    const int token = blockIdx.x, h_start = blockIdx.y * 16;
    const int lane = threadIdx.x, warp = blockIdx.z;
    const int head = h_start + lane % 16, local_head = head % 64;
    const int group      = (local_head / 16 + 4 - warp) % 4;
    const int64_t stride = int64_t(a.H) * 516;
    const float* base = a.partial_ptr + int64_t(token) * Splits * stride + (head / 64) * 64 * 516;
    float maxima[Entries], sums[Entries], factors[Entries];
    float maximum = -__FLT_MAX__;
    static_for<Entries>([&](auto i) {
        const int s  = i.value * 4 + lane / 16;
        const auto x = *reinterpret_cast<const vector_t<float, 4>*>(
            base + (s < Splits ? s : 0) * stride + 64 * 512 + local_head * 4);
        sums[i.value]   = s < Splits ? x[1] : 0.0f;
        maxima[i.value] = x[0];
        maximum         = fmaxf(maximum, sums[i.value] > 0.0f ? x[0] : -__FLT_MAX__);
    });
    maximum           = fmaxf(maximum, __shfl_xor(maximum, 16));
    maximum           = fmaxf(maximum, __shfl_xor(maximum, 32));
    const float sink  = a.attn_sink_ptr
                            ? static_cast<const float*>(a.attn_sink_ptr)[head] * 1.44269504089f
                            : -__builtin_inff();
    maximum           = fmaxf(maximum, sink);
    float denominator = 0.0f;
    static_for<Entries>([&](auto i) {
        factors[i.value] =
            sums[i.value] > 0.0f ? __builtin_amdgcn_exp2f(maxima[i.value] - maximum) : 0.0f;
        denominator += sums[i.value] * factors[i.value];
    });
    denominator += __shfl_xor(denominator, 16);
    denominator += __shfl_xor(denominator, 32);
    if(a.attn_sink_ptr)
        denominator += __builtin_amdgcn_exp2f(sink - maximum);
    const float reciprocal = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    vector_t<float, 32> output{};
    static_for<Splits>([&](auto s) {
        const float factor = __shfl(factors[s.value / 4], lane % 16 + (s.value % 4) * 16);
        static_for<8>([&](auto i) {
            const auto x = *reinterpret_cast<const vector_t<float, 4>*>(
                base + s.value * stride + (group * 8 + i.value) * 1024 + (warp * 64 + lane) * 4);
            static_for<4>([&](auto j) { output[i.value * 4 + j.value] += x[j.value] * factor; });
        });
    });
    output *= reciprocal;
    a.debug_ptr = nullptr;
    a.max_ptr   = nullptr;
    a.lse_ptr   = nullptr;
    a.out_ptr   = nullptr;
    fused_epilogue<T, decltype(output), true>(a, output, warp, lane, token, h_start);
#endif
}

template <bool WriteLSE = false>
__global__
    __launch_bounds__(64) void fused_h16_tiled_split_reduce(FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    const int lane = threadIdx.x, token = blockIdx.x;
    const int head = blockIdx.y * 4 + lane % 4, d = blockIdx.z * 64 + (lane / 4) * 4;
    constexpr int stride = 16 * 516;
    const float* base    = a.partial_ptr + int64_t(token) * a.num_splits * stride;
    float sums[2], maxima[2], factors[2];
    float maximum = -__FLT_MAX__;
    static_for<2>([&](auto i) {
        const int split  = lane / 4 + i.value * 16;
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(
            base + (split < a.num_splits ? split : 0) * stride + 16 * 512 + head * 4);
        sums[i.value]   = split < a.num_splits ? stats[1] : 0.0f;
        maxima[i.value] = stats[0];
        maximum         = fmaxf(maximum, sums[i.value] > 0.0f ? stats[0] : -__FLT_MAX__);
    });
    for(int offset = 4; offset < 64; offset *= 2)
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset));
    if constexpr(WriteLSE)
    {
        if(blockIdx.z == 0)
        {
            float attention_sum = 0.0f;
            static_for<2>([&](auto i) {
                if(sums[i.value] > 0.0f)
                    attention_sum +=
                        sums[i.value] * __builtin_amdgcn_exp2f(maxima[i.value] - maximum);
            });
            for(int offset = 4; offset < 64; offset *= 2)
                attention_sum += __shfl_xor(attention_sum, offset);
            if(lane < 4)
                static_cast<float*>(a.lse_ptr)[int64_t(token) * 16 + head] =
                    attention_sum > 0.0f
                        ? (maximum + __builtin_amdgcn_logf(attention_sum)) * 0.69314718056f
                        : __builtin_inff();
        }
    }
    const float sink  = a.attn_sink_ptr
                            ? static_cast<const float*>(a.attn_sink_ptr)[head] * 1.44269504089f
                            : -__builtin_inff();
    maximum           = fmaxf(maximum, sink);
    float denominator = 0.0f;
    static_for<2>([&](auto i) {
        factors[i.value] =
            sums[i.value] > 0.0f ? __builtin_amdgcn_exp2f(maxima[i.value] - maximum) : 0.0f;
        denominator += sums[i.value] * factors[i.value];
    });
    for(int offset = 4; offset < 64; offset *= 2)
        denominator += __shfl_xor(denominator, offset);
    if(a.attn_sink_ptr)
        denominator += __builtin_amdgcn_exp2f(sink - maximum);
    const float reciprocal = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    vector_t<float, 4> output{};
    const int warp        = d / 128;
    const int source_lane = head + (d % 16) / 4 * 16;
    const int chunk       = (d % 128) / 16;
    for(int split = 0; split < a.num_splits; split += 4)
    {
        vector_t<float, 4> values[4];
        float weights[4];
        static_for<4>([&](auto i) {
            const int index      = split + i.value;
            const int safe_index = index < a.num_splits ? index : split;
            const float factor   = safe_index < 16 ? factors[0] : factors[1];
            weights[i.value] =
                index < a.num_splits ? __shfl(factor, (safe_index % 16) * 4 + lane % 4) : 0.0f;
            values[i.value] = *reinterpret_cast<const vector_t<float, 4>*>(
                base + safe_index * stride + chunk * 1024 + (warp * 64 + source_lane) * 4);
        });
        static_for<4>([&](auto i) {
            static_for<4>(
                [&](auto j) { output[j.value] += values[i.value][j.value] * weights[i.value]; });
        });
    }
    output = cast<float>(cast<dtypes::bf16>(output * reciprocal));
    if(d >= 448)
    {
        const int64_t position = a.positions64 ? static_cast<const int64_t*>(a.positions_ptr)[token]
                                               : static_cast<const int*>(a.positions_ptr)[token];
        const auto* cosine =
            static_cast<const dtypes::bf16*>(a.cos_ptr) + position * 32 + (d - 448) / 2;
        const auto* sine =
            static_cast<const dtypes::bf16*>(a.sin_ptr) + position * 32 + (d - 448) / 2;
        static_for<2>([&](auto i) {
            constexpr int j = i.value * 2;
            const float c = cast<float>(cosine[i.value]), v = cast<float>(sine[i.value]);
            const float even = output[j], odd = output[j + 1];
            output[j]     = even * c + odd * v;
            output[j + 1] = odd * c - even * v;
        });
    }
    float maximum_abs = 1e-8f;
    static_for<4>([&](auto i) { maximum_abs = fmaxf(maximum_abs, fabsf(output[i.value])); });
    for(int offset = 4; offset < 32; offset *= 2)
        maximum_abs = fmaxf(maximum_abs, __shfl_xor(maximum_abs, offset));
    const unsigned bits    = std::bit_cast<unsigned>(maximum_abs / 448.0f);
    const int exponent     = int((bits - 1) >> 23) + 1 - 127;
    const float multiplier = std::bit_cast<float>(unsigned(127 - exponent) << 23);
    const auto quantized   = cast<opus::fp8_t>(output * multiplier);
    const int64_t row      = int64_t(token) * 16 + head;
    *reinterpret_cast<vector_t<opus::fp8_t, 4>*>(a.output_q_ptr + row * 512 + d) = quantized;
    if(d % 32 == 0)
        a.output_s_ptr[row * 16 + d / 32] = exponent + 127;
#endif
}

template <int Heads, int Splits = 4, bool WriteLSE = false, int HeadTile = Heads>
__global__ __launch_bounds__(64) void fused_decode_wave_reduce(FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    using namespace fused_norm_rope_attn_rope_cast_fwd;
    specialize_heads<Heads>(a);
    a.num_splits                 = Splits;
    a.debug_ptr                  = nullptr;
    auto* lse                    = a.lse_ptr;
    a.lse_ptr                    = nullptr;
    a.max_ptr                    = nullptr;
    a.out_ptr                    = nullptr;
    using T                      = FusedNormRopeAttnRopeCastConfig<false, 0, Heads, HeadTile>;
    constexpr int partial_groups = HeadTile / 16;
    const int token = blockIdx.x, h_start = blockIdx.y * 16;
    const int lane = threadIdx.x, warp = blockIdx.z;
    const int source_tid   = warp * 64 + lane;
    const int head         = h_start + lane % 16;
    const int head_group   = blockIdx.y % partial_groups;
    const int group        = (head_group + partial_groups - warp % partial_groups) % partial_groups;
    const int local_head   = head_group * 16 + lane % 16;
    constexpr float LOG2_E = 1.44269504089f;
    constexpr int per_group = O_ELEMS<T> / PV_GROUPS<T>;
    vector_t<float, per_group> output{};
    const auto* base =
        a.partial_ptr + (int64_t(token) * Splits * Heads + h_start / HeadTile * HeadTile) * 516;
    constexpr int64_t split_stride = int64_t(Heads) * 516;
    float maximum                  = -__FLT_MAX__;
    for(int s = 0; s < Splits; ++s)
    {
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(
            base + s * split_stride + HeadTile * 512 + local_head * 4);
        if(stats[1] > 0.0f)
            maximum = fmaxf(maximum, stats[0]);
    }
    if constexpr(WriteLSE)
    {
        if(warp == 0 && lane < 16)
        {
            float attention_sum = 0.0f;
            for(int s = 0; s < Splits; ++s)
            {
                const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(
                    base + s * split_stride + HeadTile * 512 + local_head * 4);
                if(stats[1] > 0.0f)
                    attention_sum += stats[1] * __builtin_amdgcn_exp2f(stats[0] - maximum);
            }
            static_cast<float*>(lse)[int64_t(token) * Heads + head] =
                attention_sum > 0.0f
                    ? (maximum + __builtin_amdgcn_logf(attention_sum)) * 0.69314718056f
                    : __builtin_inff();
        }
    }
    const float sink = a.attn_sink_ptr ? static_cast<const float*>(a.attn_sink_ptr)[head] * LOG2_E
                                       : -__builtin_inff();
    const float final_max = fmaxf(maximum, sink);
    float denominator     = 0.0f;
    for(int s = 0; s < Splits; ++s)
    {
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(
            base + s * split_stride + HeadTile * 512 + local_head * 4);
        const float factor = stats[1] > 0.0f ? __builtin_amdgcn_exp2f(stats[0] - final_max) : 0.0f;
        denominator += stats[1] * factor;
        static_for<per_group / 4>([&](auto i) {
            const int chunk   = group * (per_group / 4) + i.value;
            const auto values = *reinterpret_cast<const vector_t<float, 4>*>(
                base + s * split_stride + chunk * 256 * 4 + source_tid * 4);
            static_for<4>(
                [&](auto j) { output[i.value * 4 + j.value] += values[j.value] * factor; });
        });
    }
    if(a.attn_sink_ptr)
        denominator += __builtin_amdgcn_exp2f(sink - final_max);
    const float reciprocal = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    output *= reciprocal;
    fused_epilogue<T, decltype(output), true>(a, output, warp, lane, token, h_start);
#endif
}

bool use_decode_wave_reduce(const FusedNormRopeAttnRopeCastParams& a)
{
    // Four channel slices per wave avoid LDS reduction on small split grids.
    const int64_t blocks = int64_t(a.N) * a.H / 16;
    const bool supported_splits =
        a.num_splits == 4 || (a.H == 32 && (a.num_splits == 2 || a.num_splits == 3 ||
                                            a.num_splits == 5 || a.num_splits == 7));
    return (a.H == 32 || a.H == 64) && supported_splits && blocks >= 32 && blocks < 256 &&
           !a.debug_ptr && !a.max_ptr && !a.out_ptr;
}

template <bool WriteLSE>
hipError_t launch_decode_wave_reduce_impl(const FusedNormRopeAttnRopeCastParams& a,
                                          hipStream_t stream)
{
    const dim3 grid(a.N, a.H / 16, 4);
    if(a.H == 32)
    {
        switch(a.num_splits)
        {
        case 2: fused_decode_wave_reduce<32, 2, WriteLSE><<<grid, 64, 0, stream>>>(a); break;
        case 3: fused_decode_wave_reduce<32, 3, WriteLSE><<<grid, 64, 0, stream>>>(a); break;
        case 5: fused_decode_wave_reduce<32, 5, WriteLSE><<<grid, 64, 0, stream>>>(a); break;
        case 7: fused_decode_wave_reduce<32, 7, WriteLSE><<<grid, 64, 0, stream>>>(a); break;
        default: fused_decode_wave_reduce<32, 4, WriteLSE><<<grid, 64, 0, stream>>>(a); break;
        }
    }
    else
        fused_decode_wave_reduce<64, 4, WriteLSE><<<grid, 64, 0, stream>>>(a);
    return hipGetLastError();
}

hipError_t launch_decode_wave_reduce(const FusedNormRopeAttnRopeCastParams& a, hipStream_t stream)
{
    return a.lse_ptr ? launch_decode_wave_reduce_impl<true>(a, stream)
                     : launch_decode_wave_reduce_impl<false>(a, stream);
}

bool use_fused_split_reduce(const FusedNormRopeAttnRopeCastParams& a)
{
    if(use_decode_wave_reduce(a))
        return true;
    const int64_t head_rows = int64_t(a.N) * a.H;
    return !a.debug_ptr && !a.lse_ptr && !a.max_ptr && !a.out_ptr &&
           ((a.num_splits == 4 && head_rows <= 256) || (a.num_splits == 8 && head_rows <= 2048) ||
            ((a.num_splits == 16 || a.num_splits == 32) && head_rows <= 1024));
}

hipError_t launch_fused_split_reduce(const FusedNormRopeAttnRopeCastParams& a, hipStream_t stream)
{
    if(use_decode_wave_reduce(a))
        return launch_decode_wave_reduce(a, stream);
    const dim3 grid(a.N, a.H);
    if(a.num_splits == 8 && int64_t(a.N) * a.H > 256)
        fused_norm_rope_attn_rope_cast_reduce_parallel_kernel<8>
            <<<dim3(a.N, a.H / 16, 4), 64, 0, stream>>>(a);
    else
        switch(a.num_splits)
        {
        case 4:
            fused_norm_rope_attn_rope_cast_reduce_head_kernel<4><<<grid, 128, 0, stream>>>(a);
            break;
        case 8:
            fused_norm_rope_attn_rope_cast_reduce_head_kernel<8><<<grid, 128, 0, stream>>>(a);
            break;
        case 16:
            fused_norm_rope_attn_rope_cast_reduce_head_kernel<16><<<grid, 128, 0, stream>>>(a);
            break;
        case 32:
            fused_norm_rope_attn_rope_cast_reduce_head_kernel<32><<<grid, 128, 0, stream>>>(a);
            break;
        default: return hipErrorInvalidValue;
        }
    return hipGetLastError();
}

template <int SplitWarps>
__global__
__launch_bounds__(SplitWarps *
                  64) void fused_h16_parallel_split_reduce(FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    using namespace fused_norm_rope_attn_rope_cast_fwd;
    specialize_heads<16>(a);
    a.debug_ptr         = nullptr;
    a.lse_ptr           = nullptr;
    a.max_ptr           = nullptr;
    a.out_ptr           = nullptr;
    constexpr int Items = 32;
    __shared__ __align__(16) float scratch[SplitWarps * (32 + 64 * Items)];
    float* maxima  = scratch;
    float* sums    = maxima + SplitWarps * 16;
    float* values  = sums + SplitWarps * 16;
    const int lane = threadIdx.x % 64, warp = threadIdx.x / 64;
    const int token = blockIdx.x, output_warp = blockIdx.y;
    const int head = lane % 16, tid = output_warp * 64 + lane;
    const int64_t stride = 16 * 516;
    const float* base    = a.partial_ptr + int64_t(token) * a.num_splits * stride;
    float maximum        = -__FLT_MAX__;
    for(int s = warp; s < a.num_splits; s += SplitWarps)
    {
        const auto stats =
            *reinterpret_cast<const vector_t<float, 4>*>(base + s * stride + 16 * 512 + head * 4);
        if(stats[1] > 0.0f)
            maximum = fmaxf(maximum, stats[0]);
    }
    if(lane < 16)
        maxima[head * SplitWarps + warp] = maximum;
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    static_for<SplitWarps>(
        [&](auto w) { maximum = fmaxf(maximum, maxima[head * SplitWarps + w.value]); });
    const float sink      = a.attn_sink_ptr
                                ? static_cast<const float*>(a.attn_sink_ptr)[head] * 1.44269504089f
                                : -__builtin_inff();
    const float final_max = fmaxf(maximum, sink);
    vector_t<float, Items> output{};
    float denominator = 0.0f;
    for(int s = warp; s < a.num_splits; s += SplitWarps)
    {
        const auto stats =
            *reinterpret_cast<const vector_t<float, 4>*>(base + s * stride + 16 * 512 + head * 4);
        const float factor = stats[1] > 0.0f ? __builtin_amdgcn_exp2f(stats[0] - final_max) : 0.0f;
        denominator += stats[1] * factor;
        static_for<Items / 4>([&](auto i) {
            const auto v = *reinterpret_cast<const vector_t<float, 4>*>(
                base + s * stride + i.value * 256 * 4 + tid * 4);
            static_for<4>([&](auto j) { output[i.value * 4 + j.value] += v[j.value] * factor; });
        });
    }
    if(lane < 16)
        sums[head * SplitWarps + warp] = denominator;
    static_for<Items>(
        [&](auto i) { values[(i.value * SplitWarps + warp) * 64 + lane] = output[i.value]; });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    if(warp != 0)
        return;
    denominator = 0.0f;
    static_for<SplitWarps>([&](auto w) { denominator += sums[head * SplitWarps + w.value]; });
    if(a.attn_sink_ptr)
        denominator += __builtin_amdgcn_exp2f(sink - final_max);
    const float reciprocal = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    static_for<Items>([&](auto i) {
        float value = 0.0f;
        static_for<SplitWarps>(
            [&](auto w) { value += values[(i.value * SplitWarps + w.value) * 64 + lane]; });
        output[i.value] = value * reciprocal;
    });
    using T = FusedNormRopeAttnRopeCastConfig<false, 0, 16, 32>;
    fused_epilogue<T, decltype(output), true>(a, output, output_warp, lane, token, 0);
#endif
}

hipError_t launch_small_head_output_reduce(const FusedNormRopeAttnRopeCastParams& args,
                                           hipStream_t stream)
{
    if(use_decode_wave_reduce(args))
        return launch_decode_wave_reduce(args, stream);
    if(args.H == 16)
        if(args.num_splits >= 4 && int64_t(args.N) * 4 <= 256)
        {
            if(args.num_splits >= 8)
                fused_h16_parallel_split_reduce<4><<<dim3(args.N, 4), 256, 0, stream>>>(args);
            else
                fused_h16_parallel_split_reduce<2><<<dim3(args.N, 4), 128, 0, stream>>>(args);
        }
        else
            fused_norm_rope_attn_rope_cast_decode_reduce_kernel<16, 0, true, 32>
                <<<dim3(args.N), 256, 0, stream>>>(args);
    else if(args.num_splits == 2)
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<32, 2, true, 32>
            <<<dim3(args.N, 2), 256, 0, stream>>>(args);
    else if(args.num_splits == 4)
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<32, 4, true, 32>
            <<<dim3(args.N, 2), 256, 0, stream>>>(args);
    else
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<32, 0, true, 32>
            <<<dim3(args.N, 2), 256, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_small_head_full_reduce(const FusedNormRopeAttnRopeCastParams& args,
                                         hipStream_t stream)
{
    if(args.H == 16)
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<16, 0, false, 32>
            <<<dim3(args.N), 256, 0, stream>>>(args);
    else if(args.H == 32)
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<32, 0, false, 32>
            <<<dim3(args.N, 2), 256, 0, stream>>>(args);
    else
        return hipErrorInvalidValue;
    return hipGetLastError();
}

hipError_t launch_h16_tiled_split_reduce(const FusedNormRopeAttnRopeCastParams& args,
                                         hipStream_t stream)
{
    if(args.lse_ptr)
        fused_h16_tiled_split_reduce<true><<<dim3(args.N, 4, 8), 64, 0, stream>>>(args);
    else
        fused_h16_tiled_split_reduce<false><<<dim3(args.N, 4, 8), 64, 0, stream>>>(args);
    return hipGetLastError();
}

namespace {

template <int Heads>
hipError_t launch_decode_reduce_heads(const FusedNormRopeAttnRopeCastParams& args,
                                      hipStream_t stream)
{
    if(use_fused_split_reduce(args))
        return launch_fused_split_reduce(args, stream);
    if(args.num_splits == 2)
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<Heads, 2>
            <<<dim3(args.N, Heads / 16), 256, 0, stream>>>(args);
    else if(args.num_splits == 4)
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<Heads, 4>
            <<<dim3(args.N, Heads / 16), 256, 0, stream>>>(args);
    else if(!args.debug_ptr && !args.lse_ptr && !args.max_ptr && !args.out_ptr)
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<Heads, 0, true>
            <<<dim3(args.N, Heads / 16), 256, 0, stream>>>(args);
    else
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<Heads>
            <<<dim3(args.N, Heads / 16), 256, 0, stream>>>(args);
    return hipGetLastError();
}

template <int Splits>
hipError_t launch_h64_head_tile32_reduce_impl(const FusedNormRopeAttnRopeCastParams& args,
                                              hipStream_t stream)
{
    if(args.lse_ptr)
        fused_decode_wave_reduce<64, Splits, true, 32><<<dim3(args.N, 4, 4), 64, 0, stream>>>(args);
    else
        fused_decode_wave_reduce<64, Splits, false, 32>
            <<<dim3(args.N, 4, 4), 64, 0, stream>>>(args);
    return hipGetLastError();
}

} // namespace

hipError_t launch_decode_reduce(const FusedNormRopeAttnRopeCastParams& args, hipStream_t stream)
{
    if(args.H == 64)
        return launch_decode_reduce_heads<64>(args, stream);
    if(args.H == 128)
        return launch_decode_reduce_heads<128>(args, stream);
    return hipErrorInvalidValue;
}

hipError_t launch_h64_head_tile32_reduce(const FusedNormRopeAttnRopeCastParams& args,
                                         hipStream_t stream)
{
    if(args.num_splits == 4)
        return launch_h64_head_tile32_reduce_impl<4>(args, stream);
    if(args.num_splits == 5)
        return launch_h64_head_tile32_reduce_impl<5>(args, stream);
    return hipErrorInvalidValue;
}
