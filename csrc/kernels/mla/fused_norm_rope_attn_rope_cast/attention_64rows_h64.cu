// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// H64 instantiation and dispatch. H128 is compiled separately to shorten cold builds.
#include "attention_64rows.cuh"

extern template hipError_t run_fused_norm_rope_attn_rope_cast_heads<128>(
    const FusedNormRopeAttnRopeCastParams& args, bool raw_q, hipStream_t stream);

hipError_t run_fused_norm_rope_attn_rope_cast_fwd(const FusedNormRopeAttnRopeCastParams& args,
                                                  bool raw_q,
                                                  hipStream_t stream)
{
    if(args.H == 16 || args.H == 32)
        return run_fused_norm_rope_attn_rope_cast_fwd_small_heads(args, raw_q, stream);
    if(args.H == 64)
        return run_fused_norm_rope_attn_rope_cast_heads<64>(args, raw_q, stream);
    if(args.H == 128)
        return run_fused_norm_rope_attn_rope_cast_heads<128>(args, raw_q, stream);
    return hipErrorInvalidValue;
}

template <int Heads, int Splits = 4>
__global__ __launch_bounds__(64) void fused_decode_wave_reduce(
    FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    using namespace fused_norm_rope_attn_rope_cast_fwd;
    specialize_heads<Heads>(a);
    a.num_splits = Splits;
    a.debug_ptr = nullptr;
    a.lse_ptr   = nullptr;
    a.max_ptr   = nullptr;
    a.out_ptr   = nullptr;
    using T = FusedNormRopeAttnRopeCastConfig<0, false, false, 0, Heads, Heads>;
    constexpr int partial_groups = Heads / 16;
    const int token = blockIdx.x, h_start = blockIdx.y * 16;
    const int lane = threadIdx.x, warp = blockIdx.z;
    const int source_tid = warp * 64 + lane;
    const int head         = h_start + lane % 16;
    const int head_group   = blockIdx.y % partial_groups;
    const int group        = (head_group + partial_groups - warp % partial_groups) % partial_groups;
    const int local_head   = head_group * 16 + lane % 16;
    constexpr float LOG2_E = 1.44269504089f;
    constexpr int per_group = O_ELEMS<T> / PV_GROUPS<T>;
    vector_t<float, per_group> output{};
    const auto* base = a.partial_ptr + int64_t(token) * Splits * Heads * 516;
    constexpr int64_t split_stride = int64_t(Heads) * 516;
    float maximum = -__FLT_MAX__;
    for(int s = 0; s < Splits; ++s)
    {
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(base + s * split_stride +
                                                                        Heads * 512 + local_head * 4);
        if(stats[1] > 0.0f)
            maximum = fmaxf(maximum, stats[0]);
    }
    const float sink = a.attn_sink_ptr ? static_cast<const float*>(a.attn_sink_ptr)[head] * LOG2_E
                                       : -__builtin_inff();
    const float final_max = fmaxf(maximum, sink);
    float denominator = 0.0f;
    for(int s = 0; s < Splits; ++s)
    {
        const auto stats   = *reinterpret_cast<const vector_t<float, 4>*>(base + s * split_stride +
                                                                        Heads * 512 + local_head * 4);
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
    const bool supported_splits = a.num_splits == 4 ||
                                  (a.H == 32 && (a.num_splits == 3 || a.num_splits == 5 ||
                                                 a.num_splits == 7));
    return (a.H == 32 || a.H == 64) && supported_splits && blocks >= 32 && blocks < 256 &&
           !a.debug_ptr && !a.lse_ptr && !a.max_ptr && !a.out_ptr;
}

hipError_t launch_decode_wave_reduce(const FusedNormRopeAttnRopeCastParams& a, hipStream_t stream)
{
    const dim3 grid(a.N, a.H / 16, 4);
    if(a.H == 32)
    {
        switch(a.num_splits)
        {
        case 3:
            fused_decode_wave_reduce<32, 3><<<grid, 64, 0, stream>>>(a);
            break;
        case 5:
            fused_decode_wave_reduce<32, 5><<<grid, 64, 0, stream>>>(a);
            break;
        case 7:
            fused_decode_wave_reduce<32, 7><<<grid, 64, 0, stream>>>(a);
            break;
        default:
            fused_decode_wave_reduce<32><<<grid, 64, 0, stream>>>(a);
            break;
        }
    }
    else
        fused_decode_wave_reduce<64><<<grid, 64, 0, stream>>>(a);
    return hipGetLastError();
}

bool use_fused_split_reduce(const FusedNormRopeAttnRopeCastParams& a)
{
    if(use_decode_wave_reduce(a))
        return true;
    const int64_t head_rows = int64_t(a.N) * a.H;
    return !a.debug_ptr && !a.lse_ptr && !a.max_ptr && !a.out_ptr &&
           ((a.num_splits == 4 && head_rows <= 256) ||
            (a.num_splits == 8 && head_rows <= 2048) ||
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
            fused_norm_rope_attn_rope_cast_reduce_head_kernel<4>
                <<<grid, 128, 0, stream>>>(a);
            break;
        case 8:
            fused_norm_rope_attn_rope_cast_reduce_head_kernel<8>
                <<<grid, 128, 0, stream>>>(a);
            break;
        case 16:
            fused_norm_rope_attn_rope_cast_reduce_head_kernel<16>
                <<<grid, 128, 0, stream>>>(a);
            break;
        case 32:
            fused_norm_rope_attn_rope_cast_reduce_head_kernel<32>
                <<<grid, 128, 0, stream>>>(a);
            break;
        default: return hipErrorInvalidValue;
        }
    return hipGetLastError();
}

template <int SplitWarps>
__global__ __launch_bounds__(SplitWarps * 64) void fused_h16_parallel_split_reduce(
    FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    using namespace fused_norm_rope_attn_rope_cast_fwd;
    specialize_heads<16>(a);
    a.debug_ptr = nullptr;
    a.lse_ptr = nullptr;
    a.max_ptr = nullptr;
    a.out_ptr = nullptr;
    constexpr int Items = 32;
    __shared__ __align__(16) float scratch[SplitWarps * (32 + 64 * Items)];
    float* maxima = scratch;
    float* sums = maxima + SplitWarps * 16;
    float* values = sums + SplitWarps * 16;
    const int lane = threadIdx.x % 64, warp = threadIdx.x / 64;
    const int token = blockIdx.x, output_warp = blockIdx.y;
    const int head = lane % 16, tid = output_warp * 64 + lane;
    const int64_t stride = 16 * 516;
    const float* base = a.partial_ptr + int64_t(token) * a.num_splits * stride;
    float maximum = -__FLT_MAX__;
    for(int s = warp; s < a.num_splits; s += SplitWarps)
    {
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(
            base + s * stride + 16 * 512 + head * 4);
        if(stats[1] > 0.0f)
            maximum = fmaxf(maximum, stats[0]);
    }
    if(lane < 16)
        maxima[head * SplitWarps + warp] = maximum;
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    static_for<SplitWarps>([&](auto w) {
        maximum = fmaxf(maximum, maxima[head * SplitWarps + w.value]);
    });
    const float sink = a.attn_sink_ptr
                           ? static_cast<const float*>(a.attn_sink_ptr)[head] * 1.44269504089f
                           : -__builtin_inff();
    const float final_max = fmaxf(maximum, sink);
    vector_t<float, Items> output{};
    float denominator = 0.0f;
    for(int s = warp; s < a.num_splits; s += SplitWarps)
    {
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(
            base + s * stride + 16 * 512 + head * 4);
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
    static_for<Items>([&](auto i) {
        values[(i.value * SplitWarps + warp) * 64 + lane] = output[i.value];
    });
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
        static_for<SplitWarps>([&](auto w) {
            value += values[(i.value * SplitWarps + w.value) * 64 + lane];
        });
        output[i.value] = value * reciprocal;
    });
    using T = FusedNormRopeAttnRopeCastConfig<0, false, false, 0, 16, 32>;
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
