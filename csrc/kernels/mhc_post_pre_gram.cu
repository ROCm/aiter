// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// M64 decode: compute projection and Gram statistics from the rounded residual
// streams, then use those statistics to normalize their weighted sum.
// The Gram sufficient-statistic technique has public prior art:
// https://github.com/local-inference-lab/b12x/commit/3c2d08fc
// https://github.com/flashinfer-ai/flashinfer/tree/fc8fdc17/flashinfer/experimental/b12x/norm/mhc
// This HIP implementation is not a claim of algorithmic novelty.
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "mhc.h"

// Keep the gfx950 wave64 instructions out of other device compilation passes.
#if defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)
namespace mhc_decode_kparallel {
using namespace aiter;
using B         = opus::bf16_t;
using F4        = opus::vector_t<float, 4>;
constexpr int M = 64, H = 7168, HC = 4, N = 24, LD = 48, TM = 16;

__device__ __forceinline__ float row_sum(float value, int lane)
{
    int bits = __builtin_bit_cast(int, value);
    value += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, bits));
    bits = __builtin_bit_cast(int, value);
    value += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, bits));
    return value;
}

template <int SPLITS>
__global__ __launch_bounds__(256, 1) void produce(float* __restrict__ partials,
                                                  B* __restrict__ next_residual,
                                                  const B* __restrict__ x,
                                                  const B* __restrict__ residual,
                                                  const float* __restrict__ post,
                                                  const float* __restrict__ comb,
                                                  const float* __restrict__ fn)
{
    using opus::operator""_I;
    static_assert(SPLITS == 56 || SPLITS == 112);
    static_assert(H % (SPLITS * 4 * 4) == 0);
    static_assert(opus::get_warp_size() == 64);
    constexpr int K_PER_WAVE = H / (SPLITS * 4);
    constexpr int VEC        = K_PER_WAVE / 4;
    static_assert(VEC == 4 || VEC == 8);
    using BV = opus::vector_t<B, VEC>;
    using FV = opus::vector_t<float, VEC>;
    __shared__ float s_projection[3 * 64 * 8];
    __shared__ float s_gram[3 * TM * 10];

    const int lane  = threadIdx.x % 64;
    const int wave  = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int row0  = blockIdx.x * TM;
    const int split = blockIdx.y;
    const int row   = lane % TM;
    const int k     = split * (H / SPLITS) + wave * K_PER_WAVE + lane / TM * VEC;
    auto gx         = opus::make_gmem<B>(const_cast<B*>(x + row0 * H), TM * H * sizeof(B));
    auto gr = opus::make_gmem<B>(const_cast<B*>(residual + row0 * HC * H), TM * HC * H * sizeof(B));
    auto gn = opus::make_gmem<B>(next_residual + row0 * HC * H, TM * HC * H * sizeof(B));
    auto gf = opus::make_gmem<float>(const_cast<float*>(fn), N * HC * H * sizeof(float));
    auto go = opus::make_gmem<float>(partials + (split * M + row0) * LD, TM * LD * sizeof(float));

    // All four input streams are read once for this wave's own H interval.
    BV xv = load<VEC>(gx, row * H + k);
    BV rv[HC];
#pragma unroll
    for(int h = 0; h < HC; ++h)
        rv[h] = load<VEC>(gr, row * HC * H + h * H + k);

    // Keep rounded output streams in registers. Each stream is stored once
    // by its unique wave/Kgroup and reused by both projection and Gram.
    FV rounded[HC];
    BV rounded_bf16[HC];
#pragma unroll
    for(int dest = 0; dest < HC; ++dest)
    {
        const float pv = post[(row0 + row) * HC + dest];
        float cv[HC];
#pragma unroll
        for(int source = 0; source < HC; ++source)
            cv[source] = comb[(row0 + row) * HC * HC + source * HC + dest];
        BV stored;
#pragma unroll
        for(int e = 0; e < VEC; ++e)
        {
            float value = static_cast<float>(xv[e]) * pv;
#pragma unroll
            for(int source = 0; source < HC; ++source)
                value += static_cast<float>(rv[source][e]) * cv[source];
            stored[e]        = opus::fp32_to_bf16(value);
            rounded[dest][e] = static_cast<float>(stored[e]);
        }
        rounded_bf16[dest] = stored;
        store<VEC>(gn, stored, row * HC * H + dest * H + k);
    }

    // Ten upper-triangle Gram products. Cross-lane sums then complete this
    // wave's K interval for every row without any workgroup synchronization.
    float gram[10];
#pragma unroll
    for(int a = 0; a < HC; ++a)
    {
#pragma unroll
        for(int b = a; b < HC; ++b)
        {
            const int field = a * (9 - a) / 2 + b - a;
            float value     = 0.0f;
#pragma unroll
            for(int e = 0; e < VEC; ++e)
                value += rounded[a][e] * rounded[b][e];
            gram[field] = row_sum(value, lane);
        }
    }

    F4 projection[2] = {F4{0, 0, 0, 0}, F4{0, 0, 0, 0}};
#pragma unroll
    for(int head = 0; head < HC; ++head)
    {
        // Native MFMA layout: lane%16 selects N for fn and M for rounded;
        // lane/16 supplies one of four K slices. C's lane-to-M,N map is
        // unchanged by making waves partition H instead of residual heads.
        const int off = (lane % TM) * HC * H + head * H + k;
        FV weights[2];
        weights[0] = load_vector_nbytes<float, VEC, 16, 0, false>(gf, off);
        weights[1] = load_vector_nbytes<float, VEC, 16, 0, false>(gf, off + TM * HC * H);
#pragma unroll
        for(int n = 0; n < 2; ++n)
        {
            BV hi, lo;
#pragma unroll
            for(int e = 0; e < VEC; ++e)
            {
                hi[e] = opus::fp32_to_bf16(weights[n][e]);
                lo[e] = opus::fp32_to_bf16(weights[n][e] - static_cast<float>(hi[e]));
            }
            if constexpr(VEC == 8)
            {
                projection[n] =
                    opus::mfma_f32_16x16x32_bf16{}(hi, rounded_bf16[head], projection[n]);
                projection[n] =
                    opus::mfma_f32_16x16x32_bf16{}(lo, rounded_bf16[head], projection[n]);
            }
            else
            {
                projection[n] =
                    opus::mfma_f32_16x16x16_bf16{}(hi, rounded_bf16[head], projection[n]);
                projection[n] =
                    opus::mfma_f32_16x16x16_bf16{}(lo, rounded_bf16[head], projection[n]);
            }
        }
    }

    // The only block barrier: three waves deposit their projection and Gram
    // contributions; wave0 completes both sums and uniquely owns stores.
    if(wave != 0)
    {
#pragma unroll
        for(int n = 0; n < 2; ++n)
        {
#pragma unroll
            for(int e = 0; e < 4; ++e)
                s_projection[((wave - 1) * 64 + lane) * 8 + n * 4 + e] = projection[n][e];
        }
        if(lane < TM)
        {
#pragma unroll
            for(int g = 0; g < 10; ++g)
                s_gram[((wave - 1) * TM + lane) * 10 + g] = gram[g];
        }
    }
    s_wait_all_dscnt(0_I);
    __builtin_amdgcn_s_barrier();
    if(wave == 0)
    {
#pragma unroll
        for(int w = 0; w < 3; ++w)
        {
#pragma unroll
            for(int n = 0; n < 2; ++n)
            {
#pragma unroll
                for(int e = 0; e < 4; ++e)
                    projection[n][e] += s_projection[(w * 64 + lane) * 8 + n * 4 + e];
            }
        }
        const int off = row * LD + lane / TM * 4;
        store<4>(go, projection[0], off);
        store<4>(go, projection[1], off + TM);
        if(lane < TM)
        {
#pragma unroll
            for(int g = 0; g < 10; ++g)
            {
                float value = gram[g];
#pragma unroll
                for(int w = 0; w < 3; ++w)
                    value += s_gram[(w * TM + lane) * 10 + g];
                partials[(split * M + row0 + lane) * LD + 32 + g] = value;
            }
        }
    }
}

} // namespace mhc_decode_kparallel

namespace mhc_decode_finish {
using namespace aiter;
using B         = opus::bf16_t;
using B8        = opus::vector_t<B, 8>;
constexpr int M = 64, H = 7168, HC = 4, LD = 48;
constexpr int FIELDS = 40, GROUPS = 8, THREADS = FIELDS * GROUPS;
// Empirical cancellation policy, not a proven bound on floating-point error.
constexpr float CANCELLATION_RATIO = 1.0e-3f;

template <int LANE>
__device__ __forceinline__ float lane_value(float value)
{
    return __builtin_bit_cast(float,
                              __builtin_amdgcn_readlane(__builtin_bit_cast(int, value), LANE));
}

__device__ __forceinline__ float sum_row(float value, int lane)
{
    (void)lane;
    value += opus::mov_dpp(value, opus::number<0xb1>{});
    value += opus::mov_dpp(value, opus::number<0x4e>{});
    return value;
}

__device__ __forceinline__ float sum_col(float value, int lane)
{
    (void)lane;
    // These are the native cross-row DPP rotations within a16-lane row.
    value += opus::mov_dpp(value, opus::number<0x124>{});
    value += opus::mov_dpp(value, opus::number<0x128>{});
    return value;
}

__device__ __forceinline__ float max_row(float value, int lane)
{
    (void)lane;
    value = fmaxf(value, opus::mov_dpp(value, opus::number<0xb1>{}));
    value = fmaxf(value, opus::mov_dpp(value, opus::number<0x4e>{}));
    return value;
}

__device__ __forceinline__ float sigmoid(float value) { return 1.0f / (1.0f + __expf(-value)); }

template <int SPLITS, int PARTS>
__global__ __launch_bounds__(THREADS, 1) void finish(float* __restrict__ post,
                                                     float* __restrict__ comb,
                                                     B* __restrict__ output,
                                                     float* __restrict__ partials,
                                                     const B* __restrict__ residual,
                                                     const float* __restrict__ scale,
                                                     const float* __restrict__ base,
                                                     const B* __restrict__ norm_weight,
                                                     float rms_eps,
                                                     float pre_eps,
                                                     float sink_eps,
                                                     float norm_eps,
                                                     float post_mult)
{
    using opus::operator""_I;
    static_assert(SPLITS == 56 || SPLITS == 112);
    static_assert(PARTS == 1 || PARTS == 4);
    static_assert(opus::get_warp_size() == 64);
    constexpr int SEGMENT      = H / PARTS;
    constexpr int VECS         = SEGMENT / 8;
    constexpr int FOLD_THREADS = 256;
    static_assert(SEGMENT % 8 == 0);
    __shared__ float s_reduction[GROUPS * FIELDS];
    __shared__ float s_pre[4];
    __shared__ float s_inverse_norm;
    __shared__ float s_logits[20];
    __shared__ int s_direct_norm;

    const int row   = blockIdx.x;
    const int part  = blockIdx.y;
    const int tid   = threadIdx.x;
    const int lane  = tid % 64;
    const int wave  = __builtin_amdgcn_readfirstlane(tid / 64);
    const int field = tid % FIELDS;
    const int group = tid / FIELDS;
    // Logical fields0..23 are projection;24..33 are the ten Gram entries.
    // Neither physical padding range is ever read, even speculatively.
    const int physical_field = field < 24 ? field : field + 8;
    float acc[4]             = {0.0f, 0.0f, 0.0f, 0.0f};
    if(field < 34)
    {
#pragma unroll
        for(int first = group; first < SPLITS; first += GROUPS * 4)
        {
            float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for(int u = 0; u < 4; ++u)
            {
                const int split = first + u * GROUPS;
                if(split < SPLITS)
                    values[u] = partials[(split * M + row) * LD + physical_field];
            }
#pragma unroll
            for(int u = 0; u < 4; ++u)
                acc[u] += values[u];
        }
    }
    s_reduction[tid] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    s_wait_all_dscnt(0_I);
    __builtin_amdgcn_s_barrier();

    if(wave == 0)
    {
        float stat = 0.0f;
        if(lane < FIELDS)
        {
            // Eight split-group sums with four independent accumulation
            // chains. Every reduction slot was initialized above.
            float a0 = s_reduction[lane] + s_reduction[4 * FIELDS + lane];
            float a1 = s_reduction[FIELDS + lane] + s_reduction[5 * FIELDS + lane];
            float a2 = s_reduction[2 * FIELDS + lane] + s_reduction[6 * FIELDS + lane];
            float a3 = s_reduction[3 * FIELDS + lane] + s_reduction[7 * FIELDS + lane];
            stat     = (a0 + a1) + (a2 + a3);
        }
        // All readlane operations execute with the entire first wave active.
        const float g00                = lane_value<24>(stat);
        const float g01                = lane_value<25>(stat);
        const float g02                = lane_value<26>(stat);
        const float g03                = lane_value<27>(stat);
        const float g11                = lane_value<28>(stat);
        const float g12                = lane_value<29>(stat);
        const float g13                = lane_value<30>(stat);
        const float g22                = lane_value<31>(stat);
        const float g23                = lane_value<32>(stat);
        const float g33                = lane_value<33>(stat);
        const float trace              = g00 + g11 + g22 + g33;
        const float inv_projection_rms = rsqrtf(trace / (HC * H) + rms_eps);
        float logit                    = 0.0f;
        if(lane < 24)
        {
            const int which_scale = lane < 4 ? 0 : lane < 8 ? 1 : 2;
            logit                 = stat * inv_projection_rms * scale[which_scale] + base[lane];
        }
        const float pre = lane < 4 ? sigmoid(logit) + pre_eps : 0.0f;
        const float p0  = lane_value<0>(pre);
        const float p1  = lane_value<1>(pre);
        const float p2  = lane_value<2>(pre);
        const float p3  = lane_value<3>(pre);
        // In real arithmetic p^T G p equals the sum of squared folded values.
        // Small/nonfinite quadratics require a direct norm; no value is clamped.
        const float squared_norm = p0 * p0 * g00 + p1 * p1 * g11 + p2 * p2 * g22 + p3 * p3 * g33 +
                                   2.0f * (p0 * p1 * g01 + p0 * p2 * g02 + p0 * p3 * g03 +
                                           p1 * p2 * g12 + p1 * p3 * g13 + p2 * p3 * g23);
        const float diagonal = p0 * p0 * g00 + p1 * p1 * g11 + p2 * p2 * g22 + p3 * p3 * g33;
        if(lane < 4)
            s_pre[lane] = pre;
        if(lane == 0)
        {
            const float argument  = squared_norm / H + norm_eps;
            const float threshold = CANCELLATION_RATIO * diagonal;
            s_direct_norm = !__builtin_isfinite(squared_norm) || !__builtin_isfinite(diagonal) ||
                            !__builtin_isfinite(argument) || argument <= 0.0f ||
                            squared_norm <= threshold;
            if(!s_direct_norm)
                s_inverse_norm = rsqrtf(argument);
#ifdef MHC_GRAM_FALLBACK_DIAGNOSTICS
            // Test-only build: production emits none of these stores.
            if(part == 0)
            {
                partials[row * LD + 42] = static_cast<float>(s_direct_norm);
                partials[row * LD + 43] = squared_norm;
                partials[row * LD + 44] = diagonal;
                partials[row * LD + 45] = argument;
                partials[row * LD + 46] = squared_norm;
                partials[row * LD + 47] = threshold;
            }
#endif
        }
        if(lane >= 4 && lane < 24)
            s_logits[lane - 4] = logit;
    }
    s_wait_all_dscnt(0_I);
    __builtin_amdgcn_s_barrier();

    if(s_direct_norm)
    {
        // Match the upstream M64 norm's coordinate assignment and reduction:
        // 256 threads, four adjacent heads, seven 1024-element blocks. Each
        // PARTS CTA recomputes the complete row, avoiding cross-CTA barriers.
        auto sum_f = [](float a, float b) { return a + b; };
        if(tid < FOLD_THREADS)
        {
            const int head    = tid % HC;
            const int h_group = tid / HC;
            const float pre   = s_pre[head];
            auto gr =
                opus::make_gmem<B>(const_cast<B*>(residual + row * HC * H), HC * H * sizeof(B));
            float sumsq = 0.0f;
            for(int i = 0; i < H / 1024; ++i)
            {
                auto values =
                    load_vector_nbytes<B, 16, 16, 0, false>(gr, head * H + i * 1024 + h_group * 16);
#pragma unroll
                for(int e = 0; e < 16; ++e)
                {
                    float product = static_cast<float>(values[e]) * pre;
                    float folded  = multithread_reduce(product, sum_f, HC);
                    if(head == 0)
                        sumsq += folded * folded;
                }
            }
            sumsq = wave_reduce<float, decltype(sum_f), 64, false>(sumsq, sum_f);
            if(lane == 63)
                s_reduction[wave] = sumsq;
        }
        s_wait_all_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
        if(wave == 0)
        {
            float sumsq = s_reduction[lane % 4];
            sumsq       = wave_reduce<float, decltype(sum_f), 4, true>(sumsq, sum_f);
            if(lane == 0)
            {
                s_inverse_norm = rsqrtf(sumsq / H + norm_eps);
#ifdef MHC_GRAM_FALLBACK_DIAGNOSTICS
                if(part == 0)
                    partials[row * LD + 46] = sumsq;
#endif
            }
        }
        s_wait_all_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    // With the norm ready, output and Sinkhorn no longer need block barriers.
    if(tid < FOLD_THREADS)
    {
        const float p0 = s_pre[0], p1 = s_pre[1];
        const float p2 = s_pre[2], p3 = s_pre[3];
        const float inv_norm = s_inverse_norm;
        auto gr = opus::make_gmem<B>(const_cast<B*>(residual + row * HC * H), HC * H * sizeof(B));
        auto gw = opus::make_gmem<B>(const_cast<B*>(norm_weight), H * sizeof(B));
        auto go = opus::make_gmem<B>(output + row * H, H * sizeof(B));
        if(s_direct_norm)
        {
            // Recompute the owned segment with the upstream four-head tree.
            // Upstream normalizes a BF16-rounded fold using the unrounded norm.
            static_assert(SEGMENT % FOLD_THREADS == 0);
            auto sum_f        = [](float a, float b) { return a + b; };
            const int head    = tid % HC;
            const int h_group = tid / HC;
            for(int i = 0; i < SEGMENT / FOLD_THREADS; ++i)
            {
                const int h = part * SEGMENT + i * FOLD_THREADS + h_group * 4;
                auto values = load<4>(gr, head * H + h);
                opus::vector_t<B, 4> rounded;
#pragma unroll
                for(int e = 0; e < 4; ++e)
                {
                    float product = static_cast<float>(values[e]) * s_pre[head];
                    float folded  = multithread_reduce(product, sum_f, HC);
                    rounded[e]    = opus::cast<B>(folded);
                }
                if(head == 0)
                {
                    auto w = load<4>(gw, h);
                    opus::vector_t<B, 4> out;
#pragma unroll
                    for(int e = 0; e < 4; ++e)
                        out[e] = opus::cast<B>(static_cast<float>(rounded[e]) *
                                               static_cast<float>(w[e]) * inv_norm);
                    store<4>(go, out, h);
                }
            }
        }
        else
        {
#pragma unroll
            for(int v = tid; v < VECS; v += FOLD_THREADS)
            {
                const int h = part * SEGMENT + v * 8;
                B8 r0       = load<8>(gr, h);
                B8 r1       = load<8>(gr, H + h);
                B8 r2       = load<8>(gr, 2 * H + h);
                B8 r3       = load<8>(gr, 3 * H + h);
                B8 w        = load<8>(gw, h);
                B8 out;
#pragma unroll
                for(int e = 0; e < 8; ++e)
                {
                    const float folded =
                        static_cast<float>(r0[e]) * p0 + static_cast<float>(r1[e]) * p1 +
                        static_cast<float>(r2[e]) * p2 + static_cast<float>(r3[e]) * p3;
                    out[e] = opus::fp32_to_bf16(folded * inv_norm * static_cast<float>(w[e]));
                }
                store<8>(go, out, h);
            }
        }
    }
    else if(part == 0)
    {
        // Only partition0 owns post/comb. Wave4 lanes0..15 hold the4x4
        // combination matrix in row-major order; exchanges stay in each
        // 16-lane group, so inactive outputs cannot affect these lanes.
        if(lane < 4)
            post[row * HC + lane] = sigmoid(s_logits[lane]) * post_mult;
        float value = lane < 16 ? s_logits[4 + lane] : 0.0f;
        value       = __expf(value - max_row(value, lane));
        value       = value / sum_row(value, lane) + sink_eps;
        value       = value / (sum_col(value, lane) + sink_eps);
#pragma unroll
        for(int iteration = 0; iteration < 19; ++iteration)
        {
            value *= __builtin_amdgcn_rcpf(sum_row(value, lane) + sink_eps);
            value *= __builtin_amdgcn_rcpf(sum_col(value, lane) + sink_eps);
        }
        if(lane < 16)
            comb[row * HC * HC + lane] = value;
    }
}

} // namespace mhc_decode_finish

#endif

namespace aiter {
namespace {
void check_mhc_gram_tensor(const aiter_tensor_t& t,
                           AiterDtype dtype,
                           int device,
                           std::initializer_list<int64_t> shape)
{
    AITER_CHECK(t.is_gpu() && t.device_id == device && t.dtype() == dtype,
                "mHC Gram: tensor device/dtype mismatch");
    AITER_CHECK(t.dim() == static_cast<int>(shape.size()) && t.is_contiguous(),
                "mHC Gram: expected contiguous tensor of declared rank");
    int dim = 0;
    for(const auto size : shape)
        AITER_CHECK(t.size(dim++) == size, "mHC Gram: tensor shape mismatch");
    AITER_CHECK(reinterpret_cast<uintptr_t>(t.ptr) % 16 == 0,
                "mHC Gram: tensor pointer must be 16-byte aligned");
}
} // namespace

void mhc_post_pre_gram_gfx950(aiter_tensor_t& post,
                              aiter_tensor_t& comb,
                              aiter_tensor_t& out,
                              aiter_tensor_t& next_residual,
                              aiter_tensor_t& partials,
                              aiter_tensor_t& layer_input,
                              aiter_tensor_t& residual_in,
                              aiter_tensor_t& post_mix,
                              aiter_tensor_t& comb_mix,
                              aiter_tensor_t& fn,
                              aiter_tensor_t& hc_scale,
                              aiter_tensor_t& hc_base,
                              aiter_tensor_t& weight,
                              float rms_eps,
                              float pre_eps,
                              float sink_eps,
                              float norm_eps,
                              float post_mult)
{
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(layer_input.is_gpu(), "mHC Gram requires GPU inputs");
    int current_device = -1;
    HIP_CALL(hipGetDevice(&current_device));
    AITER_CHECK(current_device == layer_input.device_id,
                "mHC Gram: set the current device to the tensor device before binding its stream");
    const HipDeviceGuard guard(layer_input.device_id);
    AITER_CHECK(get_gpu_arch() == "gfx950", "mHC Gram requires gfx950");
    AITER_CHECK(get_num_cu_func() == 256 && getenv("HSA_CU_MASK") == nullptr &&
                    getenv("ROC_GLOBAL_CU_MASK") == nullptr,
                "mHC Gram tuning requires an unmasked 256-CU device");
    constexpr int m = 64, hidden = 7168, splits = 56;
    const int device = layer_input.device_id;
    check_mhc_gram_tensor(layer_input, AITER_DTYPE_bf16, device, {m, hidden});
    check_mhc_gram_tensor(residual_in, AITER_DTYPE_bf16, device, {m, 4, hidden});
    check_mhc_gram_tensor(next_residual, AITER_DTYPE_bf16, device, {m, 4, hidden});
    check_mhc_gram_tensor(post_mix, AITER_DTYPE_fp32, device, {m, 4});
    check_mhc_gram_tensor(comb_mix, AITER_DTYPE_fp32, device, {m, 4, 4});
    check_mhc_gram_tensor(fn, AITER_DTYPE_fp32, device, {24, 4 * hidden});
    check_mhc_gram_tensor(hc_scale, AITER_DTYPE_fp32, device, {3});
    check_mhc_gram_tensor(hc_base, AITER_DTYPE_fp32, device, {24});
    check_mhc_gram_tensor(weight, AITER_DTYPE_bf16, device, {hidden});
    check_mhc_gram_tensor(post, AITER_DTYPE_fp32, device, {m, 4, 1});
    check_mhc_gram_tensor(comb, AITER_DTYPE_fp32, device, {m, 4, 4});
    check_mhc_gram_tensor(out, AITER_DTYPE_bf16, device, {m, hidden});
    check_mhc_gram_tensor(partials, AITER_DTYPE_fp32, device, {splits, m, 48});
#if defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)
    using B           = opus::bf16_t;
    const auto stream = getCurrentHIPStream();
    mhc_decode_kparallel::produce<56>
        <<<dim3(4, 56), 256, 0, stream>>>(static_cast<float*>(partials.ptr),
                                          static_cast<B*>(next_residual.ptr),
                                          static_cast<const B*>(layer_input.ptr),
                                          static_cast<const B*>(residual_in.ptr),
                                          static_cast<const float*>(post_mix.ptr),
                                          static_cast<const float*>(comb_mix.ptr),
                                          static_cast<const float*>(fn.ptr));
    HIP_CALL_LAUNCH(hipGetLastError());
    mhc_decode_finish::finish<56, 4>
        <<<dim3(64, 4), 320, 0, stream>>>(static_cast<float*>(post.ptr),
                                          static_cast<float*>(comb.ptr),
                                          static_cast<B*>(out.ptr),
                                          static_cast<float*>(partials.ptr),
                                          static_cast<const B*>(next_residual.ptr),
                                          static_cast<const float*>(hc_scale.ptr),
                                          static_cast<const float*>(hc_base.ptr),
                                          static_cast<const B*>(weight.ptr),
                                          rms_eps,
                                          pre_eps,
                                          sink_eps,
                                          norm_eps,
                                          post_mult);
    HIP_CALL_LAUNCH(hipGetLastError());
#endif
}
} // namespace aiter
