// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Decode route of mhc_fused_post_pre on gfx950 (selected in aiter/ops/mhc.py), two launches:
//   produce  the post step: the four new residual streams, rounded to BF16 and stored,
//            their projection onto fn, and the ten entries of their Gram matrix G (the
//            inner products of the stream pairs). One workgroup handles 16 rows and one
//            slab of the hidden dimension and stores partial sums.
//   finish   adds the partial sums of a row, forms the pre, post and comb mixes, folds
//            the streams with the pre mix p and applies RMSNorm. The squared norm of
//            the fold is p^T G p, so a row is normalised without a pass over its fold,
//            and several workgroups can each write a segment of the same row.
// The Gram formulation has public prior art:
// https://github.com/local-inference-lab/b12x/commit/3c2d08fc
// https://github.com/flashinfer-ai/flashinfer/tree/fc8fdc17/flashinfer/experimental/b12x/norm/mhc
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "mhc.h"

// The kernel bodies are compiled in the gfx950 device pass. The host pass and every
// other device pass get empty kernels, so that the launch symbols resolve there.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#define MHC_GRAM_GFX950 1
#else
#define MHC_GRAM_GFX950 0
#endif

namespace aiter {
namespace mhc_gram {
using B = opus::bf16_t;
// Residual streams, fn rows, rows of a producer tile. A row has LD floats of partial
// sums per split: the projection in 0..23 (24..31 are the rows past fn of the second
// 16-row MFMA band, zero), the Gram entries from GRAM on.
constexpr int HC = 4, N = 24, TM = 16, LD = 48, GRAM = 32;
// The finisher: FIELDS x GROUPS threads add the partial sums, FOLD_THREADS of them fold.
constexpr int FIELDS = 40, GROUPS = 8, THREADS = FIELDS * GROUPS, FOLD_THREADS = 256;

#if MHC_GRAM_GFX950
using F4 = opus::vector_t<float, 4>;
using F8 = opus::vector_t<float, 8>;
using B8 = opus::vector_t<B, 8>;
// The Gram norm is used while p^T G p exceeds this share of its diagonal part.
// An empirical policy for cancelling streams, not a bound on the rounding error.
constexpr float CANCELLATION_RATIO = 1.0e-3f;

// Sum over the four lanes that hold one row of a 16-row tile.
__device__ __forceinline__ float tile_row_sum(float value, int lane)
{
    int bits = __builtin_bit_cast(int, value);
    value += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, bits));
    bits = __builtin_bit_cast(int, value);
    value += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, bits));
    return value;
}
#endif

// A lane holds VEC hidden elements of its row per step: 8 (16x16x32 MFMA) or 4
// (16x16x16). PACKED: fn is the BF16 hi/lo layout of mhc_shuffle_fn; otherwise it is
// FP32 and split into hi and lo here.
// HIDDEN and SPLITS are the hidden size and the split count at compile time for the
// configurations of the table, so that the offsets and the step count fold; 0 takes
// the kernel argument.
template <int VEC, bool PACKED, int HIDDEN, int SPLITS>
__global__ __launch_bounds__(256, 1) void produce(float* __restrict__ partials,
                                                  B* __restrict__ next_residual,
                                                  const B* __restrict__ x,
                                                  const B* __restrict__ residual,
                                                  const float* __restrict__ post,
                                                  const float* __restrict__ comb,
                                                  const float* __restrict__ fn,
                                                  int m,
                                                  int hidden_arg,
                                                  int splits_arg)
{
#if MHC_GRAM_GFX950
    using opus::operator""_I;
    static_assert(VEC == 4 || VEC == 8);
    static_assert(opus::get_warp_size() == 64);
    // Cache policy of the loads of x and the residual streams, which are read once per
    // call: plain loads measured faster for hidden 7168, non-temporal loads for hidden
    // 4096 (MI350X, inputs in the cache and rotating inputs). fn is read by every row
    // tile and keeps plain loads.
    constexpr auto ACT_AUX = HIDDEN == 7168 ? RT : GROUP_NT;
    const int hidden       = HIDDEN ? HIDDEN : hidden_arg;
    const int splits       = SPLITS ? SPLITS : splits_arg;
    using BV               = opus::vector_t<B, VEC>;
    using FV               = opus::vector_t<float, VEC>;
    __shared__ float s_projection[3 * 64 * 8];
    __shared__ float s_gram[3 * TM * 10];

    const int lane  = threadIdx.x % 64;
    const int wave  = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int row0  = blockIdx.x * TM;
    const int split = blockIdx.y;
    const int row   = lane % TM;
    const int full  = HC * hidden;
    // A last tile of fewer than 16 rows: the buffers end at row m, so the other rows
    // load zeros and store nothing. Their mixes are read from the tile's last row.
    const int rows    = m - row0 < TM ? m - row0 : TM;
    const int mix_row = row0 + (row < rows ? row : rows - 1);
    const int64_t r0  = row0;
    auto gx = opus::make_gmem<B>(const_cast<B*>(x + r0 * hidden), rows * hidden * (int)sizeof(B));
    auto gr =
        opus::make_gmem<B>(const_cast<B*>(residual + r0 * full), rows * full * (int)sizeof(B));
    auto gn = opus::make_gmem<B>(next_residual + r0 * full, rows * full * (int)sizeof(B));
    auto gf = opus::make_gmem<float>(const_cast<float*>(fn), N * full * (int)sizeof(float));
    float* const tile_partials = partials + (static_cast<int64_t>(split) * m + r0) * LD;
    auto go = opus::make_gmem<float>(tile_partials, rows * LD * (int)sizeof(float));

    float post_v[HC];
    float comb_v[HC][HC];
#pragma unroll
    for(int dest = 0; dest < HC; ++dest)
    {
        post_v[dest] = post[mix_row * HC + dest];
#pragma unroll
        for(int source = 0; source < HC; ++source)
            comb_v[dest][source] = comb[mix_row * HC * HC + source * HC + dest];
    }

    float gram[10]   = {};
    F4 projection[2] = {F4{0, 0, 0, 0}, F4{0, 0, 0, 0}};
    // A step takes 16 * VEC hidden elements of the slab: a quarter per wave, VEC per lane.
    const int steps = hidden / (splits * 16 * VEC);
    for(int step = 0; step < steps; ++step)
    {
        const int k = split * (hidden / splits) + (step * 4 + wave) * (4 * VEC) + lane / TM * VEC;
        const BV xv = load<VEC>(gx, row * hidden + k, 0, opus::number<ACT_AUX>{});
        BV rv[HC];
#pragma unroll
        for(int h = 0; h < HC; ++h)
            rv[h] = load<VEC>(gr, row * full + h * hidden + k, 0, opus::number<ACT_AUX>{});

        // The rounded streams stay in registers: each is stored once and then used by the
        // Gram products and the projection. The sum has the operation order of mhc_post as
        // compiled for gfx950 (the product of stream 0 rounded, then x * post and streams
        // 1..3 fused), spelled out so that next_residual has the bits mhc_post stores.
        FV rounded[HC];
        BV rounded_bf16[HC];
#pragma unroll
        for(int dest = 0; dest < HC; ++dest)
        {
            BV stored;
#pragma unroll
            for(int e = 0; e < VEC; ++e)
            {
                float value = static_cast<float>(rv[0][e]) * comb_v[dest][0];
                value       = __builtin_fmaf(static_cast<float>(xv[e]), post_v[dest], value);
#pragma unroll
                for(int h = 1; h < HC; ++h)
                    value = __builtin_fmaf(static_cast<float>(rv[h][e]), comb_v[dest][h], value);
                stored[e]        = opus::fp32_to_bf16(value);
                rounded[dest][e] = static_cast<float>(stored[e]);
            }
            rounded_bf16[dest] = stored;
            store<VEC>(gn, stored, row * full + dest * hidden + k);
        }

        // The ten upper-triangle products of the rounded streams, summed over the four
        // lanes of a row here: the lane exchanges then overlap the weight loads below.
        // In front of the barrier they ran one after the other.
#pragma unroll
        for(int a = 0; a < HC; ++a)
        {
#pragma unroll
            for(int b = a; b < HC; ++b)
            {
                float value = 0.0f;
#pragma unroll
                for(int e = 0; e < VEC; ++e)
                    value += rounded[a][e] * rounded[b][e];
                gram[a * (9 - a) / 2 + b - a] += tile_row_sum(value, lane);
            }
        }

#pragma unroll
        for(int head = 0; head < HC; ++head)
        {
            // MFMA operands: lane % 16 is the fn row (two bands of 16) and the row of
            // the tile, lane / 16 the K group. fn rows 24..31 are past the buffer: zero.
            const int off = (lane % TM) * full + head * hidden + k;
            BV hi[2], lo[2];
#pragma unroll
            for(int n = 0; n < 2; ++n)
            {
                if constexpr(PACKED)
                {
                    // Eight K elements are stored as 8 hi BF16 (4 floats), then 8 lo.
                    using FH      = opus::vector_t<float, VEC / 2>;
                    const int run = off + n * TM * full - k % 8 + k % 8 / 2;
                    const FH h4   = load_vector_nbytes<float, VEC / 2, 2 * VEC, 0, false>(gf, run);
                    const FH l4 =
                        load_vector_nbytes<float, VEC / 2, 2 * VEC, 0, false>(gf, run + 4);
                    hi[n] = __builtin_bit_cast(BV, h4);
                    lo[n] = __builtin_bit_cast(BV, l4);
                }
                else
                {
                    const FV weights =
                        load_vector_nbytes<float, VEC, 16, 0, false>(gf, off + n * TM * full);
#pragma unroll
                    for(int e = 0; e < VEC; ++e)
                    {
                        hi[n][e] = opus::fp32_to_bf16(weights[e]);
                        lo[n][e] = opus::fp32_to_bf16(weights[e] - static_cast<float>(hi[n][e]));
                    }
                }
            }
#pragma unroll
            for(int n = 0; n < 2; ++n)
            {
                F4& c = projection[n];
                if constexpr(VEC == 8)
                {
                    c = opus::mfma_f32_16x16x32_bf16{}(hi[n], rounded_bf16[head], c);
                    c = opus::mfma_f32_16x16x32_bf16{}(lo[n], rounded_bf16[head], c);
                }
                else
                {
                    c = opus::mfma_f32_16x16x16_bf16{}(hi[n], rounded_bf16[head], c);
                    c = opus::mfma_f32_16x16x16_bf16{}(lo[n], rounded_bf16[head], c);
                }
            }
        }
    }

    // The only block barrier: waves 1..3 deposit their projection and Gram sums,
    // wave 0 adds them to its own and stores.
    if(wave != 0)
    {
#pragma unroll
        for(int i = 0; i < 8; ++i)
            s_projection[((wave - 1) * 64 + lane) * 8 + i] = projection[i / 4][i % 4];
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
            for(int i = 0; i < 8; ++i)
                projection[i / 4][i % 4] += s_projection[(w * 64 + lane) * 8 + i];
        }
        const int off = row * LD + lane / TM * 4;
        store<4>(go, projection[0], off);
        store<4>(go, projection[1], off + TM);
        if(lane < rows)
        {
#pragma unroll
            for(int g = 0; g < 10; ++g)
            {
                float value = gram[g];
#pragma unroll
                for(int w = 0; w < 3; ++w)
                    value += s_gram[(w * TM + lane) * 10 + g];
                tile_partials[lane * LD + GRAM + g] = value;
            }
        }
    }
#endif
}

#if MHC_GRAM_GFX950
template <int LANE>
__device__ __forceinline__ float lane_value(float value)
{
    return __builtin_bit_cast(float,
                              __builtin_amdgcn_readlane(__builtin_bit_cast(int, value), LANE));
}

// Row and column reductions of a 4x4 matrix held row-major in lanes 0..15.
__device__ __forceinline__ float sum_row(float value)
{
    value += opus::mov_dpp(value, opus::number<0xb1>{});
    return value + opus::mov_dpp(value, opus::number<0x4e>{});
}

__device__ __forceinline__ float sum_col(float value)
{
    value += opus::mov_dpp(value, opus::number<0x124>{});
    return value + opus::mov_dpp(value, opus::number<0x128>{});
}

__device__ __forceinline__ float max_row(float value)
{
    value = fmaxf(value, opus::mov_dpp(value, opus::number<0xb1>{}));
    return fmaxf(value, opus::mov_dpp(value, opus::number<0x4e>{}));
}

__device__ __forceinline__ float sigmoid(float value) { return 1.0f / (1.0f + __expf(-value)); }

// Elements h .. h + 7 of the fold of one row's four streams (streams starts at its first
// stream) with the pre mix p, in the form of the fold of mhc_pre_big_fuse_rmsnorm_kernel:
// one product per stream lane, then a pairwise reduce in which each lane fuses its own
// product into its first add. The FMAs are spelled out: when the streams cancel, implicit
// contraction changes the folded value (see the cancellation cases of op_tests/test_mhc.py).
__device__ __forceinline__ F8 fold_vector(opus::gmem<B>& streams, int hidden, int h, const float* p)
{
    const B8 r0 = load<8>(streams, h);
    const B8 r1 = load<8>(streams, hidden + h);
    const B8 r2 = load<8>(streams, 2 * hidden + h);
    const B8 r3 = load<8>(streams, 3 * hidden + h);
    F8 folded;
#pragma unroll
    for(int e = 0; e < 8; ++e)
    {
        const float f0 = static_cast<float>(r0[e]), f1 = static_cast<float>(r1[e]);
        const float f2 = static_cast<float>(r2[e]), f3 = static_cast<float>(r3[e]);
        folded[e] = __builtin_fmaf(f0, p[0], f1 * p[1]) + __builtin_fmaf(f2, p[2], f3 * p[3]);
    }
    return folded;
}
#endif

// One workgroup per row and hidden segment (parts segments per row). HIDDEN, SPLITS and
// PARTS: as in produce.
template <int HIDDEN, int SPLITS, int PARTS>
__global__ __launch_bounds__(THREADS, 1) void finish(float* __restrict__ post,
                                                     float* __restrict__ comb,
                                                     B* __restrict__ output,
                                                     const float* __restrict__ partials,
                                                     const B* __restrict__ residual,
                                                     const float* __restrict__ scale,
                                                     const float* __restrict__ base,
                                                     const B* __restrict__ norm_weight,
                                                     int m,
                                                     int hidden_arg,
                                                     int splits_arg,
                                                     int parts_arg,
                                                     float rms_eps,
                                                     float pre_eps,
                                                     float sink_eps,
                                                     float norm_eps,
                                                     float post_mult)
{
#if MHC_GRAM_GFX950
    using opus::operator""_I;
    static_assert(opus::get_warp_size() == 64);
    const int hidden = HIDDEN ? HIDDEN : hidden_arg;
    const int splits = SPLITS ? SPLITS : splits_arg;
    const int parts  = PARTS ? PARTS : parts_arg;
    __shared__ float s_reduction[GROUPS * FIELDS];
    __shared__ float s_pre[HC];
    __shared__ float s_inverse_norm;
    __shared__ float s_logits[N - HC];
    __shared__ int s_direct_norm;

    const int row   = blockIdx.x;
    const int part  = blockIdx.y;
    const int tid   = threadIdx.x;
    const int lane  = tid % 64;
    const int wave  = __builtin_amdgcn_readfirstlane(tid / 64);
    const int field = tid % FIELDS;
    const int group = tid / FIELDS;
    // Thread (group, field) adds one field of the row over the splits group,
    // group + GROUPS, ...: fields 0..23 are the projection, 24..33 the Gram entries.
    const float* const field_partials =
        partials + static_cast<int64_t>(row) * LD + (field < N ? field : field + GRAM - N);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if(field < N + 10)
    {
        if constexpr(SPLITS != 0)
        {
            // A fixed trip count, unrolled: every load is issued before the first sum.
            // A loop over the splits waits for its loads once per round.
            constexpr int ROUNDS = (SPLITS + GROUPS * 4 - 1) / (GROUPS * 4);
#pragma unroll
            for(int round = 0; round < ROUNDS; ++round)
            {
                const int first = group + round * GROUPS * 4;
                float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for(int u = 0; u < 4; ++u)
                {
                    const int split = first + u * GROUPS;
                    if(split < splits)
                        values[u] = field_partials[static_cast<int64_t>(split) * m * LD];
                }
#pragma unroll
                for(int u = 0; u < 4; ++u)
                    acc[u] += values[u];
            }
        }
        else
        {
            for(int first = group; first < splits; first += GROUPS * 4)
            {
                float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for(int u = 0; u < 4; ++u)
                {
                    const int split = first + u * GROUPS;
                    if(split < splits)
                        values[u] = field_partials[static_cast<int64_t>(split) * m * LD];
                }
#pragma unroll
                for(int u = 0; u < 4; ++u)
                    acc[u] += values[u];
            }
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
            // The eight group sums of this lane's field.
            const float a0 = s_reduction[lane] + s_reduction[4 * FIELDS + lane];
            const float a1 = s_reduction[FIELDS + lane] + s_reduction[5 * FIELDS + lane];
            const float a2 = s_reduction[2 * FIELDS + lane] + s_reduction[6 * FIELDS + lane];
            const float a3 = s_reduction[3 * FIELDS + lane] + s_reduction[7 * FIELDS + lane];
            stat           = (a0 + a1) + (a2 + a3);
        }
        // Every readlane runs with the whole wave active.
        const float g00 = lane_value<24>(stat), g01 = lane_value<25>(stat);
        const float g02 = lane_value<26>(stat), g03 = lane_value<27>(stat);
        const float g11 = lane_value<28>(stat), g12 = lane_value<29>(stat);
        const float g13 = lane_value<30>(stat), g22 = lane_value<31>(stat);
        const float g23 = lane_value<32>(stat), g33 = lane_value<33>(stat);
        const float inv_projection_rms = rsqrtf((g00 + g11 + g22 + g33) / (HC * hidden) + rms_eps);
        float logit                    = 0.0f;
        if(lane < N)
        {
            const int which_scale = lane < 4 ? 0 : lane < 8 ? 1 : 2;
            logit                 = stat * inv_projection_rms * scale[which_scale] + base[lane];
        }
        const float pre = lane < 4 ? sigmoid(logit) + pre_eps : 0.0f;
        const float p0 = lane_value<0>(pre), p1 = lane_value<1>(pre);
        const float p2 = lane_value<2>(pre), p3 = lane_value<3>(pre);
        // In real arithmetic p^T G p is the sum of the squared fold.
        const float diagonal = p0 * p0 * g00 + p1 * p1 * g11 + p2 * p2 * g22 + p3 * p3 * g33;
        const float squared_norm =
            diagonal + 2.0f * (p0 * p1 * g01 + p0 * p2 * g02 + p0 * p3 * g03 + p1 * p2 * g12 +
                               p1 * p3 * g13 + p2 * p3 * g23);
        if(lane < 4)
            s_pre[lane] = pre;
        if(lane == 0)
        {
            // When the streams cancel, the quadratic is a small difference of large
            // terms (or not finite): the norm is then summed from the fold itself.
            const float argument = squared_norm / hidden + norm_eps;
            const bool direct    = !__builtin_isfinite(squared_norm) ||
                                !__builtin_isfinite(diagonal) || !__builtin_isfinite(argument) ||
                                argument <= 0.0f || squared_norm <= CANCELLATION_RATIO * diagonal;
            s_direct_norm = direct;
            if(!direct)
                s_inverse_norm = rsqrtf(argument);
        }
        if(lane >= 4 && lane < N)
            s_logits[lane - 4] = logit;
    }
    s_wait_all_dscnt(0_I);
    __builtin_amdgcn_s_barrier();

    const int vectors = hidden / 8; // 16-byte vectors of a row
    const float p[HC] = {s_pre[0], s_pre[1], s_pre[2], s_pre[3]};
    auto gr = opus::make_gmem<B>(const_cast<B*>(residual + static_cast<int64_t>(row) * HC * hidden),
                                 HC * hidden * (int)sizeof(B));
    if(s_direct_norm)
    {
        // Every segment's workgroup sums the whole row, so none waits for another.
        auto sum_f  = [](float a, float b) { return a + b; };
        float sumsq = 0.0f;
        if(tid < FOLD_THREADS)
        {
            for(int v = tid; v < vectors; v += FOLD_THREADS)
            {
                const F8 folded = fold_vector(gr, hidden, v * 8, p);
#pragma unroll
                for(int e = 0; e < 8; ++e)
                    sumsq += folded[e] * folded[e];
            }
            sumsq = wave_reduce<float, decltype(sum_f), 64, false>(sumsq, sum_f);
            if(lane == 63)
                s_reduction[wave] = sumsq;
        }
        s_wait_all_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
        if(wave == 0)
        {
            sumsq = wave_reduce<float, decltype(sum_f), 4, true>(s_reduction[lane % 4], sum_f);
            if(lane == 0)
                s_inverse_norm = rsqrtf(sumsq / hidden + norm_eps);
        }
        s_wait_all_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    // With the norm known, the output and the Sinkhorn steps need no further barrier.
    if(tid < FOLD_THREADS)
    {
        const float inv_norm = s_inverse_norm;
        const int segment    = vectors / parts;
        auto gw = opus::make_gmem<B>(const_cast<B*>(norm_weight), hidden * (int)sizeof(B));
        auto go = opus::make_gmem<B>(output + static_cast<int64_t>(row) * hidden,
                                     hidden * (int)sizeof(B));
#pragma unroll
        for(int v = tid; v < segment; v += FOLD_THREADS)
        {
            const int h     = (part * segment + v) * 8;
            const F8 folded = fold_vector(gr, hidden, h, p);
            const B8 w      = load<8>(gw, h);
            B8 out;
#pragma unroll
            for(int e = 0; e < 8; ++e)
                out[e] = opus::fp32_to_bf16(folded[e] * inv_norm * static_cast<float>(w[e]));
            store<8>(go, out, h);
        }
    }
    else if(part == 0)
    {
        // The fifth wave of the row's first segment owns post and comb: lanes 0..15
        // hold the 4x4 comb matrix row-major, and the exchanges stay inside them.
        if(lane < 4)
            post[row * HC + lane] = sigmoid(s_logits[lane]) * post_mult;
        float value = lane < 16 ? s_logits[4 + lane] : 0.0f;
        value       = expf(value - max_row(value));
        value       = value / sum_row(value) + sink_eps;
        value       = value / (sum_col(value) + sink_eps);
#pragma unroll
        for(int iteration = 0; iteration < 19; ++iteration)
        {
            value *= __builtin_amdgcn_rcpf(sum_row(value) + sink_eps);
            value *= __builtin_amdgcn_rcpf(sum_col(value) + sink_eps);
        }
        if(lane < 16)
            comb[row * HC * HC + lane] = value;
    }
#endif
}
} // namespace mhc_gram

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
                              float post_mult,
                              int parts)
{
    // A failed check raises in the caller instead of ending the process.
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(layer_input.is_gpu() && layer_input.dim() == 2 && partials.dim() == 3,
                "mHC Gram: layer_input must be a (m, hidden) GPU tensor, partials (splits, m, 48)");
    const HipDeviceGuard guard(layer_input.device_id);
    static const bool is_gfx950 = get_gpu_arch() == "gfx950";
    AITER_CHECK(is_gfx950, "mHC Gram requires gfx950");
    // The kernels index one row tile or one row with 32-bit offsets.
    AITER_CHECK(layer_input.size(0) > 0 && layer_input.size(0) <= (1 << 20) &&
                    layer_input.size(1) > 0 && layer_input.size(1) <= (1 << 20) &&
                    partials.size(0) > 0 && partials.size(0) <= layer_input.size(1),
                "mHC Gram: unsupported row count, hidden size or split count");
    const int m      = static_cast<int>(layer_input.size(0));
    const int hidden = static_cast<int>(layer_input.size(1));
    const int splits = static_cast<int>(partials.size(0));
    // A producer workgroup takes hidden / splits elements: one 16x16x16 MFMA step (64)
    // or steps of 16x16x32 (128 each). A finisher workgroup takes hidden / parts.
    const int slab = hidden / splits;
    AITER_CHECK(slab * splits == hidden && (slab == 64 || slab % 128 == 0),
                "mHC Gram: hidden / splits must be 64 or a multiple of 128, got hidden ",
                hidden,
                ", splits ",
                splits);
    AITER_CHECK(parts > 0 && hidden % (8 * parts) == 0,
                "mHC Gram: hidden must be a multiple of 8 * parts, got parts ",
                parts);
    // fn is FP32 or the packed int32 layout of mhc_shuffle_fn; the kernels read every
    // tensor through its data pointer, so the element counts are what must hold.
    const bool packed = fn.dtype() == AITER_DTYPE_i32;
    const auto bf16 = AITER_DTYPE_bf16, fp32 = AITER_DTYPE_fp32;
    const int device = layer_input.device_id;
    auto is          = [&](const aiter_tensor_t& t, AiterDtype dtype, int64_t numel) {
        return t.is_gpu() && t.device_id == device && t.dtype() == dtype && t.is_contiguous() &&
               static_cast<int64_t>(t.numel()) == numel &&
               reinterpret_cast<uintptr_t>(t.ptr) % 16 == 0;
    };
    const int64_t rows = m, full = 4 * hidden;
    AITER_CHECK(is(layer_input, bf16, rows * hidden) && is(residual_in, bf16, rows * full) &&
                    is(next_residual, bf16, rows * full) && is(out, bf16, rows * hidden) &&
                    is(weight, bf16, hidden) && is(post_mix, fp32, rows * 4) &&
                    is(comb_mix, fp32, rows * 16) &&
                    is(fn, packed ? AITER_DTYPE_i32 : fp32, 24 * full) && is(hc_scale, fp32, 3) &&
                    is(hc_base, fp32, 24) && is(post, fp32, rows * 4) &&
                    is(comb, fp32, rows * 16) && is(partials, fp32, rows * splits * mhc_gram::LD),
                "mHC Gram: every tensor must be a contiguous, 16-byte aligned GPU tensor of the "
                "documented type and size");

    using B           = mhc_gram::B;
    const auto stream = getCurrentHIPStream();
    const dim3 grid((m + mhc_gram::TM - 1) / mhc_gram::TM, splits);
#define MHC_GRAM_PRODUCE(VEC, PACKED, HIDDEN, SPLITS)                       \
    mhc_gram::produce<VEC, PACKED, HIDDEN, SPLITS>                          \
        <<<grid, 256, 0, stream>>>(static_cast<float*>(partials.ptr),       \
                                   static_cast<B*>(next_residual.ptr),      \
                                   static_cast<const B*>(layer_input.ptr),  \
                                   static_cast<const B*>(residual_in.ptr),  \
                                   static_cast<const float*>(post_mix.ptr), \
                                   static_cast<const float*>(comb_mix.ptr), \
                                   static_cast<const float*>(fn.ptr),       \
                                   m,                                       \
                                   hidden,                                  \
                                   splits)
    // The configurations of MHC_GRAM_CONFIG_GFX950_256 (aiter/ops/mhc.py) run kernels with
    // their sizes at compile time; any other configuration runs the kernels that take the
    // sizes as arguments.
#define MHC_GRAM_TUNED(H, S)                                      \
    if(hidden == H && splits == S)                                \
    {                                                             \
        if(packed)                                                \
            MHC_GRAM_PRODUCE((H / S == 64 ? 4 : 8), true, H, S);  \
        else                                                      \
            MHC_GRAM_PRODUCE((H / S == 64 ? 4 : 8), false, H, S); \
    }                                                             \
    else
    MHC_GRAM_TUNED(7168, 112)
    MHC_GRAM_TUNED(7168, 56)
    MHC_GRAM_TUNED(4096, 64)
    MHC_GRAM_TUNED(4096, 32)
    MHC_GRAM_TUNED(4096, 16)
    switch(2 * packed + (slab == 64))
    {
    case 0: MHC_GRAM_PRODUCE(8, false, 0, 0); break;
    case 1: MHC_GRAM_PRODUCE(4, false, 0, 0); break;
    case 2: MHC_GRAM_PRODUCE(8, true, 0, 0); break;
    default: MHC_GRAM_PRODUCE(4, true, 0, 0);
    }
#undef MHC_GRAM_TUNED
#undef MHC_GRAM_PRODUCE
    HIP_CALL_LAUNCH(hipGetLastError());
#define MHC_GRAM_FINISH(HIDDEN, SPLITS, PARTS)                                                 \
    mhc_gram::finish<HIDDEN, SPLITS, PARTS><<<dim3(m, parts), mhc_gram::THREADS, 0, stream>>>( \
        static_cast<float*>(post.ptr),                                                         \
        static_cast<float*>(comb.ptr),                                                         \
        static_cast<B*>(out.ptr),                                                              \
        static_cast<const float*>(partials.ptr),                                               \
        static_cast<const B*>(next_residual.ptr),                                              \
        static_cast<const float*>(hc_scale.ptr),                                               \
        static_cast<const float*>(hc_base.ptr),                                                \
        static_cast<const B*>(weight.ptr),                                                     \
        m,                                                                                     \
        hidden,                                                                                \
        splits,                                                                                \
        parts,                                                                                 \
        rms_eps,                                                                               \
        pre_eps,                                                                               \
        sink_eps,                                                                              \
        norm_eps,                                                                              \
        post_mult)
#define MHC_GRAM_TUNED(H, S, P)                  \
    if(hidden == H && splits == S && parts == P) \
        MHC_GRAM_FINISH(H, S, P);                \
    else
    MHC_GRAM_TUNED(7168, 112, 4)
    MHC_GRAM_TUNED(7168, 56, 4)
    MHC_GRAM_TUNED(7168, 56, 2)
    MHC_GRAM_TUNED(4096, 64, 2)
    MHC_GRAM_TUNED(4096, 32, 2)
    MHC_GRAM_TUNED(4096, 32, 1)
    MHC_GRAM_TUNED(4096, 16, 2)
    MHC_GRAM_TUNED(4096, 16, 1)
    MHC_GRAM_FINISH(0, 0, 0);
#undef MHC_GRAM_TUNED
#undef MHC_GRAM_FINISH
    HIP_CALL_LAUNCH(hipGetLastError());
}
} // namespace aiter
