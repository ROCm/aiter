// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "mhc.h"
#include "opus/opus.hpp"

namespace aiter {
namespace mhc_gfx950 {
// Keep the gfx950 wave64 instructions out of other device compilation passes.
#if defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)
using bf4 = unsigned short __attribute__((ext_vector_type(4)));
__device__ __forceinline__ float bf_to_f(unsigned short b)
{
    return __uint_as_float(static_cast<unsigned int>(b) << 16);
}
__device__ __forceinline__ unsigned short f_to_bf(float f)
{
    unsigned int u = __float_as_uint(f);
    if((u & 0x7fffffffU) > 0x7f800000U)
        return (u >> 16) | 0x40U;
    return static_cast<unsigned short>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ float wave_sum(float x)
{
#pragma unroll
    for(int d = 32; d; d >>= 1)
        x += __shfl_xor(x, d, 64);
    return x;
}
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
template <int CONTROL>
__device__ __forceinline__ float dpp(float x)
{
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), CONTROL, 0xf, 0xf, false));
}
__device__ __forceinline__ float row_sum(float x)
{
    x += dpp<0xb1>(x);
    return x + dpp<0x4e>(x);
}
__device__ __forceinline__ float col_sum(float x)
{
    x += dpp<0x124>(x);
    return x + dpp<0x128>(x);
}

using bf8 = unsigned short __attribute__((ext_vector_type(8)));
__device__ __forceinline__ float rounded_cross_row_sum_4(float value, int lane)
{
    value += __builtin_bit_cast(
        float, __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, __builtin_bit_cast(int, value)));
    value += __builtin_bit_cast(
        float, __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, __builtin_bit_cast(int, value)));
    return value;
}

template <int TM, int TK>
__global__ __launch_bounds__(256, 1) void mhc_rounded_packed_prefill(float* out,
                                                                     float* sqrsum,
                                                                     opus::bf16_t* next_residual,
                                                                     const opus::bf16_t* x,
                                                                     const opus::bf16_t* residual,
                                                                     const float* packed_fn,
                                                                     const float* post_mix,
                                                                     const float* comb_mix,
                                                                     int m,
                                                                     int hidden_size,
                                                                     int x_stride,
                                                                     int split_k)
{
    using opus::operator""_I;
    using bf16       = opus::bf16_t;
    constexpr int HC = 4, N = 24, TN = 32;
    constexpr int WS = 64, BLOCK = 256, MM = 16, MN = 16, OVEC = 4;
    constexpr int MR = TM / MM, NR = TN / MN, VEC = TK / (WS / MM);
    constexpr int TILE = TM * TK, PV = 8, DV = 8;
    constexpr int PR = TILE / (BLOCK * PV);
    constexpr int NJ = VEC / DV, VPL = MR * NR * OVEC;
    constexpr int RF = (HC - 1) * WS * VPL + (HC - 1) * MM * MR;
    static_assert(TM == 32 && TK == 64, "supported ownership is TM32/TK64");
    static_assert(PR == 1 && TILE == BLOCK * PV, "one producer vector per thread");
    static_assert(VEC % 8 == 0, "a lane must own whole packed kb8 blocks");
    static_assert(opus::get_warp_size() == WS, "gfx950 wave64 only");

    // Input values live only in producer registers. All head waves consume the
    // same rounded bits that are stored to next_residual. Reduction scratch is
    // reused only after every rounded-tile consumer has finished its LDS reads.
    union alignas(16) Shared
    {
        bf16 rounded[HC * TILE];
        float reduction[RF];
    };
    __shared__ Shared shared;
    const int idx       = blockIdx.x * TM;
    const int split_idx = blockIdx.z;
    const int full_k    = HC * hidden_size;
    const int k_offset  = split_idx * (hidden_size / split_k);
    const int k_loop    = hidden_size / (split_k * TK);
    const int m_oob     = m - idx < TM ? m - idx : TM;
    const int lane      = threadIdx.x % WS;
    const int wave      = __builtin_amdgcn_readfirstlane(threadIdx.x / WS);
    auto gx = opus::make_gmem<bf16>(const_cast<bf16*>(x + static_cast<int64_t>(idx) * x_stride),
                                    m_oob * x_stride * (int)sizeof(bf16));
    auto gr =
        opus::make_gmem<bf16>(const_cast<bf16*>(residual + static_cast<int64_t>(idx) * full_k),
                              m_oob * full_k * (int)sizeof(bf16));
    auto gn = opus::make_gmem<bf16>(next_residual + static_cast<int64_t>(idx) * full_k,
                                    m_oob * full_k * (int)sizeof(bf16));
    auto gf =
        opus::make_gmem<float>(const_cast<float*>(packed_fn), N * full_k * (int)sizeof(float));
    auto go     = opus::make_gmem<float>(out + (static_cast<int64_t>(split_idx) * m + idx) * TN,
                                     m_oob * TN * (int)sizeof(float));
    using pvec  = opus::vector_t<bf16, PV>;
    using pfvec = opus::vector_t<float, PV>;
    using rvec  = opus::vector_t<bf16, DV>;
    using fvec  = opus::vector_t<float, VEC>;
    using ftile = opus::array<fvec, NR>;
    using avec  = opus::vector_t<float, OVEC>;

    // At TM32/TK64/PV8, tid owns row=tid/8, hidden=(tid%8)*8.
    // One set of twenty coefficients is retained for that row, and the five
    // loaded input vectors supply all four produced heads.
    const int elem         = threadIdx.x * PV;
    const int producer_row = elem / TK;
    const int producer_k   = elem % TK;
    float pc[HC];
    float cc[HC][HC];
#pragma unroll
    for(int oh = 0; oh < HC; ++oh)
    {
        pc[oh] = producer_row < m_oob ? post_mix[(idx + producer_row) * HC + oh] : 0.0f;
#pragma unroll
        for(int ih = 0; ih < HC; ++ih)
            cc[ih][oh] = producer_row < m_oob
                             ? comb_mix[(idx + producer_row) * HC * HC + ih * HC + oh]
                             : 0.0f;
    }
    auto load_fn = [&](int step) {
        ftile tile;
        // Packed kb8 keeps eight logical K positions in eight raw float words:
        // four words containing hi BF16 pairs, then four containing lo pairs.
        // Head, split, step and lane starts are all divisible by eight.
        const int base =
            (lane % MN) * full_k + wave * hidden_size + (lane / MN) * VEC + step * TK + k_offset;
#pragma unroll
        for(int n = 0; n < NR; ++n)
            tile[n] = load_vector_nbytes<float, VEC, 16, 0, false>(gf, base + n * MN * full_k);
        return tile;
    };
    // One raw tile contains five 16-byte vectors per lane. Two register slots
    // let the next tile's global loads overlap current production and MFMA.
    struct raw_tile
    {
        pvec xv;
        pvec rv[HC];
    };
    auto load_raw = [&](int step) {
        const int global_k = k_offset + step * TK + producer_k;
        raw_tile tile;
        tile.xv =
            load_vector_nbytes<bf16, PV, 16, 0, false>(gx, producer_row * x_stride + global_k);
#pragma unroll
        for(int ih = 0; ih < HC; ++ih)
            tile.rv[ih] = load_vector_nbytes<bf16, PV, 16, 0, false>(
                gr, producer_row * full_k + ih * hidden_size + global_k);
        return tile;
    };
    raw_tile raw_ring[2];
    auto produce_tile = [&](int step, auto slot_tag) {
        constexpr int SLOT = decltype(slot_tag)::value;
        const int global_k = k_offset + step * TK + producer_k;
        const pvec& xv     = raw_ring[SLOT].xv;
        const auto& rv     = raw_ring[SLOT].rv;
#pragma unroll
        for(int oh = 0; oh < HC; ++oh)
        {
            pfvec values;
#pragma unroll
            for(int e = 0; e < PV; ++e)
                values[e] = static_cast<float>(xv[e]) * pc[oh];
#pragma unroll
            for(int ih = 0; ih < HC; ++ih)
            {
#pragma unroll
                for(int e = 0; e < PV; ++e)
                    values[e] += static_cast<float>(rv[ih][e]) * cc[ih][oh];
            }
            pvec rounded;
#pragma unroll
            for(int e = 0; e < PV; ++e)
                rounded[e] = opus::fp32_to_bf16(values[e]);
            // The split and this owner partition [row,head,hidden] exactly.
            // Buffer bounds suppress stores from a possible partial row tile.
            store_vector_nbytes<bf16, bf16, PV, 16, 0, false>(
                gn, rounded, producer_row * full_k + oh * hidden_size + global_k);
            *reinterpret_cast<pvec*>(shared.rounded + oh * TILE + elem) = rounded;
        }
    };
    ftile fn_ring[2];
    fn_ring[0]  = load_fn(0);
    raw_ring[0] = load_raw(0);
    // Current slot zero is ready before future slot one is issued. On gfx950,
    // this helper is vmcnt(0); no inference from a nonzero outstanding count is
    // needed. Scheduling barriers keep later loads on the far side of the wait.
    s_wait_all_loadcnt(0_I, 0_I);
    __builtin_amdgcn_sched_barrier(0);
    if(k_loop > 1)
    {
        fn_ring[1]  = load_fn(1);
        raw_ring[1] = load_raw(1);
    }
    __builtin_amdgcn_sched_barrier(0);
    avec accum[MR][NR];
    float sqpart[MR];
#pragma unroll
    for(int b = 0; b < MR; ++b)
    {
        sqpart[b] = 0.0f;
#pragma unroll
        for(int n = 0; n < NR; ++n)
            opus::clear(accum[b][n]);
    }
    auto consume = [&](auto slot_tag) {
        constexpr int SLOT = decltype(slot_tag)::value;
#pragma unroll
        for(int b = 0; b < MR; ++b)
        {
            const int first = wave * TILE + (b * MM + lane % MM) * TK + (lane / MM) * VEC;
#pragma unroll
            for(int j = 0; j < NJ; ++j)
            {
                const rvec rounded = *reinterpret_cast<rvec*>(shared.rounded + first + j * DV);
#pragma unroll
                for(int e = 0; e < DV; ++e)
                {
                    const float value = opus::bf16_to_fp32(rounded[e]);
                    sqpart[b] += value * value;
                }
#pragma unroll
                for(int n = 0; n < NR; ++n)
                {
                    // j=0 consumes raw words [0:4]/[4:8], and j=1 uses
                    // [8:12]/[12:16]. The BF16 activation is used verbatim.
                    using f32x4 = opus::vector_t<float, 4>;
                    f32x4 hi_raw, lo_raw;
                    constexpr int LO_OFF = 4;
                    const int off        = (j * DV / 8) * 8 + (j * DV % 8) / 2;
#pragma unroll
                    for(int e = 0; e < 4; ++e)
                    {
                        hi_raw[e] = fn_ring[SLOT][n][off + e];
                        lo_raw[e] = fn_ring[SLOT][n][off + LO_OFF + e];
                    }
                    const rvec fn_hi = __builtin_bit_cast(rvec, hi_raw);
                    const rvec fn_lo = __builtin_bit_cast(rvec, lo_raw);
                    accum[b][n]      = opus::mfma_f32_16x16x32_bf16{}(fn_hi, rounded, accum[b][n]);
                    accum[b][n]      = opus::mfma_f32_16x16x32_bf16{}(fn_lo, rounded, accum[b][n]);
                }
            }
        }
    };
    // Entry invariant for step s: raw/fn slot s%2 is ready, and the other slot
    // holds step s+1 in flight or ready (unless this is the last step). All raw
    // inputs and weights consumed by s were completed by the preceding full
    // vmcnt drain. The rounded tile needs only an LDS wait before publication;
    // the next raw/fn loads can remain in flight throughout current consumption.
    // After every wave finishes LDS reads, vmcnt(0) completes the next slot.
    // The now-dead current slot is then loaded with s+2 before advancing. Ring
    // tags stay compile-time constants, and every barrier guard is block-uniform.
    for(int first = 0; first < k_loop; first += 2)
    {
        opus::static_for<2>([&](auto slot_tag) {
            constexpr int SLOT = decltype(slot_tag)::value;
            const int step     = first + SLOT;
            if(step < k_loop)
            {
                produce_tile(step, slot_tag);
                s_wait_all_dscnt(0_I);
                __builtin_amdgcn_s_barrier();
                consume(slot_tag);
                s_wait_all_dscnt(0_I);
                __builtin_amdgcn_s_barrier();
                s_wait_all_loadcnt(0_I, 0_I);
                __builtin_amdgcn_sched_barrier(0);
                if(step + 2 < k_loop)
                {
                    fn_ring[SLOT]  = load_fn(step + 2);
                    raw_ring[SLOT] = load_raw(step + 2);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        });
    }

    // Unchanged native head reduction and [split,m,32] projection layout.
    float sqwave[MR];
#pragma unroll
    for(int b = 0; b < MR; ++b)
        sqwave[b] = rounded_cross_row_sum_4(sqpart[b], lane);
    __syncthreads();
    float* red   = shared.reduction;
    float* sqred = red + (HC - 1) * WS * VPL;
    if(wave != 0)
    {
        int c = 0;
#pragma unroll
        for(int b = 0; b < MR; ++b)
        {
#pragma unroll
            for(int n = 0; n < NR; ++n)
            {
#pragma unroll
                for(int e = 0; e < OVEC; ++e)
                    red[((wave - 1) * WS + lane) * VPL + c++] = accum[b][n][e];
            }
        }
        if(lane < MM)
        {
#pragma unroll
            for(int b = 0; b < MR; ++b)
                sqred[((wave - 1) * MM + lane) * MR + b] = sqwave[b];
        }
    }
    __syncthreads();
    if(wave == 0)
    {
#pragma unroll
        for(int h = 0; h < HC - 1; ++h)
        {
            int c = 0;
#pragma unroll
            for(int b = 0; b < MR; ++b)
            {
#pragma unroll
                for(int n = 0; n < NR; ++n)
                {
#pragma unroll
                    for(int e = 0; e < OVEC; ++e)
                        accum[b][n][e] += red[(h * WS + lane) * VPL + c++];
                }
            }
        }
#pragma unroll
        for(int b = 0; b < MR; ++b)
        {
            const int oo = (b * MM + lane % MM) * TN + (lane / MM) * OVEC;
#pragma unroll
            for(int n = 0; n < NR; ++n)
                store_vector_nbytes<float, float, OVEC, 16, 0, false>(go, accum[b][n], oo + n * MN);
            if(lane < MM && b * MM + lane < m_oob)
            {
                float value = sqwave[b];
#pragma unroll
                for(int h = 0; h < HC - 1; ++h)
                    value += sqred[(h * MM + lane) * MR + b];
                sqrsum[split_idx * m + idx + b * MM + lane] = value;
            }
        }
    }
}

template <int STATIC_SPLITS = 56>
__global__
    __launch_bounds__(320) void decode_finalize_row(const float* __restrict__ proj,
                                                    const float* __restrict__ sq,
                                                    const float* __restrict__ scale,
                                                    const float* __restrict__ base,
                                                    const unsigned short* __restrict__ residual,
                                                    const unsigned short* __restrict__ weight,
                                                    float* __restrict__ post,
                                                    float* __restrict__ comb,
                                                    unsigned short* __restrict__ out,
                                                    int splits,
                                                    float rms_eps,
                                                    float pre_eps,
                                                    float sink_eps,
                                                    float norm_eps,
                                                    float post_mult,
                                                    int repeats)
{
    constexpr int H = 7168, M = 64;
    const int row = blockIdx.x;

    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    __shared__ float projection_partial[4][24];
    __shared__ float coeff[24];
    __shared__ float residual_rstd;
    __shared__ float norm_partial[4];
    if(wave < 4)
    {
        const int group = lane & 7;
        const int s0    = wave * 8 + (lane >> 3);
        float4 v        = {0.f, 0.f, 0.f, 0.f};
        if(group < 6)
        {
            if constexpr(STATIC_SPLITS == 56)
            {
                // Same split-owner grouping as the generic reduction: waves
                // 0..2 own two splits per lane, wave 3 owns one.
                const float4 x =
                    *reinterpret_cast<const float4*>(proj + (s0 * M + row) * 32 + group * 4);
                v.x += x.x;
                v.y += x.y;
                v.z += x.z;
                v.w += x.w;
                if(wave < 3)
                {
                    const float4 y = *reinterpret_cast<const float4*>(
                        proj + ((s0 + 32) * M + row) * 32 + group * 4);
                    v.x += y.x;
                    v.y += y.y;
                    v.z += y.z;
                    v.w += y.w;
                }
            }
            else
            {
                for(int s = s0; s < splits; s += 32)
                {
                    const float4 x =
                        *reinterpret_cast<const float4*>(proj + (s * M + row) * 32 + group * 4);
                    v.x += x.x;
                    v.y += x.y;
                    v.z += x.z;
                    v.w += x.w;
                }
            }
        }
#pragma unroll
        for(int d = 8; d <= 32; d *= 2)
        {
            v.x += __shfl_xor(v.x, d, 64);
            v.y += __shfl_xor(v.y, d, 64);
            v.z += __shfl_xor(v.z, d, 64);
            v.w += __shfl_xor(v.w, d, 64);
        }
        if(lane < 6)
        {
            *reinterpret_cast<float4*>(&projection_partial[wave][group * 4]) = v;
        }
    }
    else
    {
        float ss = 0.f;
        if constexpr(STATIC_SPLITS == 56)
        {
            if(lane < 56)
                ss += sq[lane * M + row];
        }
        else
        {
            for(int s = lane; s < splits; s += 64)
                ss += sq[s * M + row];
        }
        ss = wave_sum(ss);
        if(lane == 0)
            residual_rstd = rsqrtf(ss / (4 * H) + rms_eps);
    }
    __syncthreads();
    if(t < 24)
    {
        float p = (projection_partial[0][t] + projection_partial[1][t]) +
                  (projection_partial[2][t] + projection_partial[3][t]);
        const int category = t < 4 ? 0 : (t < 8 ? 1 : 2);
        float v            = (p * residual_rstd) * scale[category] + base[t];
        coeff[t]           = t < 4 ? sigmoid(v) + pre_eps : v;
    }
    __syncthreads();

    float folded[7][4];
    if(wave < 4)
    {
        const float c0 = coeff[0], c1 = coeff[1], c2 = coeff[2], c3 = coeff[3];
        float sums[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for(int i = 0; i < 7; ++i)
        {
            const int h             = t * 4 + i * 1024;
            const unsigned short* r = residual + row * (4 * H) + h;
            const bf4 r0            = *reinterpret_cast<const bf4*>(r);
            const bf4 r1            = *reinterpret_cast<const bf4*>(r + H);
            const bf4 r2            = *reinterpret_cast<const bf4*>(r + 2 * H);
            const bf4 r3            = *reinterpret_cast<const bf4*>(r + 3 * H);
#pragma unroll
            for(int j = 0; j < 4; ++j)
            {
                float y      = bf_to_f(r0[j]) * c0 + bf_to_f(r1[j]) * c1;
                y            = y + bf_to_f(r2[j]) * c2;
                y            = y + bf_to_f(r3[j]) * c3;
                folded[i][j] = y;
                sums[j] += y * y;
            }
        }
        float sum = wave_sum((sums[0] + sums[1]) + (sums[2] + sums[3]));
        if(lane == 0)
            norm_partial[wave] = sum;
    }
    else
    {
        if(lane < 4)
            post[row * 4 + lane] = sigmoid(coeff[4 + lane]) * post_mult;
        float v  = lane < 16 ? coeff[8 + lane] : 0.f;
        float mx = fmaxf(v, dpp<0xb1>(v));
        mx       = fmaxf(mx, dpp<0x4e>(mx));
        v        = expf(v - mx);
        v        = v / row_sum(v) + sink_eps;
        v        = v / (col_sum(v) + sink_eps);
        for(int iter = 1; iter < repeats; ++iter)
        {
            v *= __builtin_amdgcn_rcpf(row_sum(v) + sink_eps);
            v *= __builtin_amdgcn_rcpf(col_sum(v) + sink_eps);
        }
        if(lane < 16)
            comb[row * 16 + lane] = v;
    }
    __syncthreads();
    if(wave < 4)
    {
        const float ss  = (norm_partial[0] + norm_partial[1]) + (norm_partial[2] + norm_partial[3]);
        const float inv = rsqrtf(ss / H + norm_eps);
#pragma unroll
        for(int i = 0; i < 7; ++i)
        {
            const int h = t * 4 + i * 1024;
            {
                bf4 w = *reinterpret_cast<const bf4*>(weight + h), y;
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    y[j] = f_to_bf((folded[i][j] * inv) * bf_to_f(w[j]));
                *reinterpret_cast<bf4*>(out + row * H + h) = y;
            }
        }
    }
}

template <int STATIC_SPLITS>
__global__
    __launch_bounds__(320) void prefill_finalize_rows(const float* __restrict__ proj,
                                                      const float* __restrict__ sq,
                                                      const float* __restrict__ scale,
                                                      const float* __restrict__ base,
                                                      const unsigned short* __restrict__ residual,
                                                      const unsigned short* __restrict__ weight,
                                                      float* __restrict__ post,
                                                      float* __restrict__ comb,
                                                      unsigned short* __restrict__ out,
                                                      int m,
                                                      int splits,
                                                      int64_t proj_split_stride,
                                                      int64_t proj_row_stride,
                                                      int64_t sq_split_stride,
                                                      float rms_eps,
                                                      float pre_eps,
                                                      float sink_eps,
                                                      float norm_eps,
                                                      float post_mult,
                                                      int repeats)
{
    constexpr int H = 7168, ROWS = 2, GROUPS = ROWS * 6;
    constexpr int RED_SLOTS = 320 / GROUPS;
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const int first_row = blockIdx.x * ROWS;
    __shared__ float projection_partial[RED_SLOTS][ROWS * 24];
    __shared__ float sq_partial[ROWS][5];
    __shared__ float coeff[ROWS][24];
    __shared__ float norm_partial[ROWS][2];

    if constexpr(STATIC_SPLITS > 0)
    {
        // One wave reduces the small split count and publishes both rows' mixes.
        // Sumsq is loaded before projection, consumed only after those loads.
        if(wave == 0)
        {
            float2 qv = {0.f, 0.f};
            if(lane < STATIC_SPLITS)
                qv = *reinterpret_cast<const float2*>(sq + lane * sq_split_stride + first_row);
            const int r = lane / 24, c = lane % 24;
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            if(lane < ROWS * 24)
            {
#pragma unroll
                for(int sp = 0; sp < STATIC_SPLITS; ++sp)
                    acc[sp % 4] +=
                        proj[sp * proj_split_stride + (first_row + r) * proj_row_stride + c];
            }
            const float q0 = wave_sum(qv.x), q1 = wave_sum(qv.y);
            if(lane < ROWS * 24)
            {
                const float p      = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                const float inv    = rsqrtf((r == 0 ? q0 : q1) / (4 * H) + rms_eps);
                const int category = c < 4 ? 0 : (c < 8 ? 1 : 2);
                const float v      = (p * inv) * scale[category] + base[c];
                coeff[r][c]        = c < 4 ? sigmoid(v) + pre_eps : v;
            }
        }
        __syncthreads();
    }
    else
    {
        // Generic distributed split reduction, retaining the native 26-slot layout
        // and four-accumulator coefficient merge. Split specialization is separate.
        float ss[ROWS] = {0.f, 0.f};
        for(int s = t; s < splits; s += 320)
        {
            const float2 v = *reinterpret_cast<const float2*>(sq + s * sq_split_stride + first_row);
            ss[0] += v.x;
            ss[1] += v.y;
        }
        if(t < RED_SLOTS * GROUPS)
        {
            const int slot = t / GROUPS, group = t % GROUPS;
            const int row = first_row + group / 6, col = (group % 6) * 4;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            for(int s = slot; s < splits; s += RED_SLOTS)
            {
                const float4 x = *reinterpret_cast<const float4*>(proj + s * proj_split_stride +
                                                                  row * proj_row_stride + col);
                v.x += x.x;
                v.y += x.y;
                v.z += x.z;
                v.w += x.w;
            }
            *reinterpret_cast<float4*>(&projection_partial[slot][group * 4]) = v;
        }
#pragma unroll
        for(int r = 0; r < ROWS; ++r)
        {
            const float sum = wave_sum(ss[r]);
            if(lane == 0)
                sq_partial[r][wave] = sum;
        }
        __syncthreads();
        if(t < ROWS * 24)
        {
            const int r = t / 24, c = t % 24;
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for(int s = 0; s < RED_SLOTS; ++s)
                acc[s % 4] += projection_partial[s][t];
            const float p = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            const float q =
                ((sq_partial[r][0] + sq_partial[r][1]) + (sq_partial[r][2] + sq_partial[r][3])) +
                sq_partial[r][4];
            const float inv    = rsqrtf(q / (4 * H) + rms_eps);
            const int category = c < 4 ? 0 : (c < 8 ? 1 : 2);
            const float v      = (p * inv) * scale[category] + base[c];
            coeff[r][c]        = c < 4 ? sigmoid(v) + pre_eps : v;
        }
        __syncthreads();
    }

    // Two feature-owner waves per row; each lane retains 7*8 FP32 numerators.
    float folded[7][8];
    if(wave < 4)
    {
        const int r = wave / 2, row = first_row + r, local_t = t % 128;
        const float c0 = coeff[r][0], c1 = coeff[r][1];
        const float c2 = coeff[r][2], c3 = coeff[r][3];
        float sums[8] = {};
#pragma unroll
        for(int i = 0; i < 7; ++i)
        {
            const int h               = local_t * 8 + i * 1024;
            const unsigned short* ptr = residual + row * (4 * H) + h;
            const bf8 r0              = *reinterpret_cast<const bf8*>(ptr);
            const bf8 r1              = *reinterpret_cast<const bf8*>(ptr + H);
            const bf8 r2              = *reinterpret_cast<const bf8*>(ptr + 2 * H);
            const bf8 r3              = *reinterpret_cast<const bf8*>(ptr + 3 * H);
#pragma unroll
            for(int j = 0; j < 8; ++j)
            {
                const float y = (bf_to_f(r0[j]) * c0 + bf_to_f(r1[j]) * c1) +
                                (bf_to_f(r2[j]) * c2 + bf_to_f(r3[j]) * c3);
                folded[i][j] = y;
                sums[j] += y * y;
            }
        }
        float sum = ((sums[0] + sums[1]) + (sums[2] + sums[3])) +
                    ((sums[4] + sums[5]) + (sums[6] + sums[7]));
        sum = wave_sum(sum);
        if(lane == 0)
            norm_partial[r][wave % 2] = sum;
    }
    else
    {
        if(lane < ROWS * 4)
        {
            const int r = lane / 4, c = lane % 4;
            post[(first_row + r) * 4 + c] = sigmoid(coeff[r][4 + c]) * post_mult;
        }
        float v  = lane < ROWS * 16 ? coeff[lane / 16][8 + lane % 16] : 0.f;
        float mx = fmaxf(v, dpp<0xb1>(v));
        mx       = fmaxf(mx, dpp<0x4e>(mx));
        v        = expf(v - mx);
        v        = v / row_sum(v) + sink_eps;
        v        = v / (col_sum(v) + sink_eps);
        for(int iter = 1; iter < repeats; ++iter)
        {
            v *= __builtin_amdgcn_rcpf(row_sum(v) + sink_eps);
            v *= __builtin_amdgcn_rcpf(col_sum(v) + sink_eps);
        }
        if(lane < ROWS * 16)
            comb[first_row * 16 + lane] = v;
    }
    __syncthreads();
    if(wave < 4)
    {
        const int r = wave / 2, row = first_row + r, local_t = t % 128;
        const float inv = rsqrtf((norm_partial[r][0] + norm_partial[r][1]) / H + norm_eps);
#pragma unroll
        for(int i = 0; i < 7; ++i)
        {
            const int h = local_t * 8 + i * 1024;
            const bf8 w = *reinterpret_cast<const bf8*>(weight + h);
            bf8 y;
#pragma unroll
            for(int j = 0; j < 8; ++j)
                y[j] = f_to_bf((folded[i][j] * inv) * bf_to_f(w[j]));
            *reinterpret_cast<bf8*>(out + row * H + h) = y;
        }
    }
}

#endif
} // namespace mhc_gfx950

namespace {
void check_tensor(const aiter_tensor_t& t,
                  AiterDtype dtype,
                  int device,
                  std::initializer_list<int64_t> shape)
{
    AITER_CHECK(t.is_gpu() && t.device_id == device && t.dtype() == dtype,
                "mHC gfx950: tensor device/dtype mismatch");
    AITER_CHECK(t.dim() == static_cast<int>(shape.size()) && t.is_contiguous(),
                "mHC gfx950: expected contiguous tensor of declared rank");
    int dim = 0;
    for(const auto size : shape)
        AITER_CHECK(t.size(dim++) == size, "mHC gfx950: tensor shape mismatch");
    AITER_CHECK(reinterpret_cast<uintptr_t>(t.ptr) % 16 == 0,
                "mHC gfx950: tensor pointer must be 16-byte aligned");
}
} // namespace

void mhc_post_project_gfx950(aiter_tensor_t& projection,
                             aiter_tensor_t& sq,
                             aiter_tensor_t& next_residual,
                             aiter_tensor_t& x,
                             aiter_tensor_t& residual,
                             aiter_tensor_t& packed_fn,
                             aiter_tensor_t& post_mix,
                             aiter_tensor_t& comb_mix)
{
    const HipDeviceGuard guard(x.device_id);
    AITER_CHECK(get_gpu_arch() == "gfx950", "mHC packed producer requires gfx950");
    constexpr int m = 8192, hidden = 7168, splits = 4;
    check_tensor(x, AITER_DTYPE_bf16, x.device_id, {m, hidden});
    check_tensor(residual, AITER_DTYPE_bf16, x.device_id, {m, 4, hidden});
    check_tensor(next_residual, AITER_DTYPE_bf16, x.device_id, {m, 4, hidden});
    check_tensor(packed_fn, AITER_DTYPE_i32, x.device_id, {24, 4 * hidden});
    check_tensor(post_mix, AITER_DTYPE_fp32, x.device_id, {m, 4});
    check_tensor(comb_mix, AITER_DTYPE_fp32, x.device_id, {m, 4, 4});
    check_tensor(projection, AITER_DTYPE_fp32, x.device_id, {splits, m, 32});
    check_tensor(sq, AITER_DTYPE_fp32, x.device_id, {splits, m});
#if defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)
    mhc_gfx950::mhc_rounded_packed_prefill<32, 64>
        <<<dim3(m / 32, 1, splits), 256, 0, getCurrentHIPStream()>>>(
            static_cast<float*>(projection.ptr),
            static_cast<float*>(sq.ptr),
            static_cast<opus::bf16_t*>(next_residual.ptr),
            static_cast<const opus::bf16_t*>(x.ptr),
            static_cast<const opus::bf16_t*>(residual.ptr),
            static_cast<const float*>(packed_fn.ptr),
            static_cast<const float*>(post_mix.ptr),
            static_cast<const float*>(comb_mix.ptr),
            m,
            hidden,
            hidden,
            splits);
    HIP_CALL_LAUNCH(hipGetLastError());
#endif
}

void mhc_finalize_gfx950(aiter_tensor_t& post,
                         aiter_tensor_t& comb,
                         aiter_tensor_t& out,
                         aiter_tensor_t& projection,
                         aiter_tensor_t& sq,
                         aiter_tensor_t& scale,
                         aiter_tensor_t& base,
                         aiter_tensor_t& residual,
                         aiter_tensor_t& weight,
                         float rms_eps,
                         float pre_eps,
                         float sink_eps,
                         float norm_eps,
                         float post_mult,
                         int repeats)
{
    const HipDeviceGuard guard(out.device_id);
    AITER_CHECK(get_gpu_arch() == "gfx950", "mHC finalizer requires gfx950");
    const int m = out.size(0), splits = m == 64 ? 56 : 4;
    AITER_CHECK((m == 64 || m == 8192) && repeats > 0,
                "mHC gfx950: unsupported row count or Sinkhorn iterations");
    check_tensor(post, AITER_DTYPE_fp32, out.device_id, {m, 4, 1});
    check_tensor(comb, AITER_DTYPE_fp32, out.device_id, {m, 4, 4});
    check_tensor(out, AITER_DTYPE_bf16, out.device_id, {m, 7168});
    check_tensor(projection, AITER_DTYPE_fp32, out.device_id, {splits, m, 32});
    check_tensor(sq, AITER_DTYPE_fp32, out.device_id, {splits, m});
    check_tensor(scale, AITER_DTYPE_fp32, out.device_id, {3});
    check_tensor(base, AITER_DTYPE_fp32, out.device_id, {24});
    check_tensor(residual, AITER_DTYPE_bf16, out.device_id, {m, 4, 7168});
    check_tensor(weight, AITER_DTYPE_bf16, out.device_id, {7168});
#if defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)
    auto* projection_ptr = static_cast<const float*>(projection.ptr);
    auto* sq_ptr         = static_cast<const float*>(sq.ptr);
    auto* scale_ptr      = static_cast<const float*>(scale.ptr);
    auto* base_ptr       = static_cast<const float*>(base.ptr);
    auto* residual_ptr   = static_cast<const unsigned short*>(residual.ptr);
    auto* weight_ptr     = static_cast<const unsigned short*>(weight.ptr);
    auto* post_ptr       = static_cast<float*>(post.ptr);
    auto* comb_ptr       = static_cast<float*>(comb.ptr);
    auto* out_ptr        = static_cast<unsigned short*>(out.ptr);
    const auto stream    = getCurrentHIPStream();
    if(m == 64)
    {
        mhc_gfx950::decode_finalize_row<56><<<m, 320, 0, stream>>>(projection_ptr,
                                                                   sq_ptr,
                                                                   scale_ptr,
                                                                   base_ptr,
                                                                   residual_ptr,
                                                                   weight_ptr,
                                                                   post_ptr,
                                                                   comb_ptr,
                                                                   out_ptr,
                                                                   splits,
                                                                   rms_eps,
                                                                   pre_eps,
                                                                   sink_eps,
                                                                   norm_eps,
                                                                   post_mult,
                                                                   repeats);
    }
    else
    {
        mhc_gfx950::prefill_finalize_rows<4><<<m / 2, 320, 0, stream>>>(projection_ptr,
                                                                        sq_ptr,
                                                                        scale_ptr,
                                                                        base_ptr,
                                                                        residual_ptr,
                                                                        weight_ptr,
                                                                        post_ptr,
                                                                        comb_ptr,
                                                                        out_ptr,
                                                                        m,
                                                                        splits,
                                                                        projection.stride(0),
                                                                        projection.stride(1),
                                                                        sq.stride(0),
                                                                        rms_eps,
                                                                        pre_eps,
                                                                        sink_eps,
                                                                        norm_eps,
                                                                        post_mult,
                                                                        repeats);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
#endif
}
} // namespace aiter
