// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm device kernels. 2D block: x = threads/row, y = rows/block.
#pragma once

// Pin fp32->bf16 to round-to-nearest-even (must precede opus.hpp).
#ifndef OPUS_FP32_to_BF16_DEFAULT
#define OPUS_FP32_to_BF16_DEFAULT 0
#endif
#include "opus/opus.hpp"
#include "opus/opus_vec_io.hpp"
#include "opus/rmsnorm_opus_quant_detail.hpp"

namespace aiter {

// Per-kernel traits carrying the element type(s) + tile consts.
template <typename Scalar, int Width, bool Gemma = false>
struct rmsnorm_opus_traits
{
    using scalar_t              = Scalar;
    static constexpr int width  = Width;
    static constexpr bool gemma = Gemma; // gemma_norm: multiply by (weight + 1)
};
template <typename In, typename Out, int Width>
struct rmsnorm_quant_opus_traits
{
    using in_t                 = In;
    using out_t                = Out;
    static constexpr int width = Width;
};
template <typename Scalar, int TileN, int RegN>
struct rmsnorm_be_opus_traits
{
    using scalar_t          = Scalar;
    static constexpr int TN = TileN;
    static constexpr int RN = RegN;
};

// rmsnorm (+ residual add when residual != 0, in-place when out == in);
// model_sensitive != 0 = T5 variant (round s*inv before *w).
template <typename Traits>
__global__ void rmsnorm_opus_kernel(void* __restrict__ out,
                                     const void* __restrict__ in,
                                     const void* __restrict__ weight,
                                     void* __restrict__ residual,
                                     float epsilon,
                                     int rows,
                                     int hidden,
                                     int model_sensitive);

// rmsnorm + dynamic/smooth quant (out int8/fp8, yscale [rows]). Pointer flags:
// residual != 0 fused-add, xscale != 0 smooth, unquant != 0 store pre-quant y.
template <typename Traits>
__global__ void rmsnorm_quant_opus(void* __restrict__ out,
                                       void* __restrict__ yscale,
                                       void* __restrict__ unquant,
                                       const void* __restrict__ in,
                                       const void* __restrict__ weight,
                                       void* __restrict__ residual,
                                       const void* __restrict__ xscale,
                                       float epsilon,
                                       int rows,
                                       int hidden,
                                       float qmax,
                                       int model_sensitive);

// Bit-exact vs CK: reproduces CK's square_sum order for its tile geometry
// (TN threads/row x RN width-8 vecs).
template <typename Traits>
__global__ void rmsnorm_be_opus(void* __restrict__ out,
                                        const void* __restrict__ in,
                                        const void* __restrict__ weight,
                                        void* __restrict__ residual,
                                        float epsilon,
                                        int rows,
                                        int hidden,
                                        int model_sensitive);

// Faithful opus port of add_rmsnorm_quant_kernel (module_rmsnorm_quant): one row
// per block. Uses the coalesced load_vector_nbytes/store_vector IO layer (opus_vec_io).
// Per-token quant (group==0) loads interleaved for coalescing; grouped quant keeps
// contiguous per-thread ownership (ILV=false) so a group of group_size elements =
// reduce_thread_size = group_size/TDS contiguous lanes. Covers no-quant, per-token &
// grouped int8/fp8/fp4, fused-add, gemma, shuffle, strided rows.
template <typename In, typename Out, int Blk, int Tds, bool Interleave>
struct arq_opus_traits
{
    using in_t                = In;
    using out_t               = Out;
    static constexpr int BLK  = Blk;
    static constexpr int TDS  = Tds;
    static constexpr bool ILV = Interleave;
};
template <typename Traits>
__global__ void add_rmsnorm_quant_opus(void* __restrict__ out,
                                       void* __restrict__ rout,
                                       void* __restrict__ scale,
                                       const void* __restrict__ in,
                                       const void* __restrict__ rin,
                                       const void* __restrict__ weight,
                                       const void* __restrict__ xscale,
                                       float epsilon,
                                       int m,
                                       int n,
                                       float qmax,
                                       int in_s,
                                       int rin_s,
                                       int rout_s,
                                       int out_s,
                                       int group_size,
                                       int shuffle,
                                       int gemma);

#if !defined(__HIP_DEVICE_COMPILE__)
// Host pass: empty stubs so the __device_stub__ symbols resolve.
template <typename Traits>
__global__ void rmsnorm_opus_kernel(void*, const void*, const void*, void*, float, int, int, int)
{
}
template <typename Traits>
__global__ void add_rmsnorm_quant_opus(void*, void*, void*, const void*, const void*, const void*,
                                       const void*, float, int, int, float, int, int, int, int, int, int, int)
{
}
template <typename Traits>
__global__ void rmsnorm_quant_opus(
    void*, void*, void*, const void*, const void*, void*, const void*, float, int, int, float, int)
{
}
template <typename Traits>
__global__ void
rmsnorm_be_opus(void*, const void*, const void*, void*, float, int, int, int)
{
}
#else
// fp32 -> quant element. int8: round-to-nearest; fp8: hardware e4m3 cvt.
template <typename out_t>
__device__ inline out_t quant_cast(float v)
{
    if constexpr(std::is_same_v<out_t, signed char>)
        return static_cast<signed char>(__builtin_rintf(v));
    else
        return opus::fp32_to_fp8(v);
}

// Per-row segmented LDS reduction; deterministic (all rows step the same strides).
template <bool IS_MAX>
__device__ inline float block_reduce(float v)
{
    __shared__ float s[1024];
    const int lane = opus::thread_id_x();
    const int tpr  = opus::block_size_x();
    const int base = opus::thread_id_y() * tpr;
    // leading barrier: reuse of s[] across two reduces races on gfx942 without it
    opus::sync_threads();
    s[base + lane] = v;
    opus::sync_threads();
    for(int stride = tpr >> 1; stride > 0; stride >>= 1)
    {
        if(lane < stride)
        {
            float o        = s[base + lane + stride];
            s[base + lane] = IS_MAX ? fmaxf(s[base + lane], o) : s[base + lane] + o;
        }
        opus::sync_threads();
    }
    return s[base];
}

// Wave64 intra-warp all-reduce via DPP (single-cycle ALU cross-lane, ~20x lower
// latency than ds_bpermute). bound_ctrl 0-fill is a valid identity here: sum operates
// on squares and max on |values|, both non-negative. Result broadcast to all lanes.
template <bool IS_MAX>
__device__ inline float warp_reduce_dpp(float v)
{
    auto op = [](float a, float b) { return IS_MAX ? fmaxf(a, b) : a + b; };
    v       = op(v, opus::mov_dpp(v, opus::number<0x111>{})); // row_shr:1
    v       = op(v, opus::mov_dpp(v, opus::number<0x112>{})); // row_shr:2
    v       = op(v, opus::mov_dpp(v, opus::number<0x114>{})); // row_shr:4
    v       = op(v, opus::mov_dpp(v, opus::number<0x118>{})); // row_shr:8
    v = op(v, opus::mov_dpp(v, opus::number<0x142>{}, opus::number<0xa>{})); // row_bcast:15
    v = op(v, opus::mov_dpp(v, opus::number<0x143>{}, opus::number<0xc>{})); // row_bcast:31
    return __builtin_bit_cast(float,
                              __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 63));
}

// Fast 1D block reduction: DPP all-reduce within a warp (wave64), then a single LDS
// exchange across warps. Replaces the LDS reduction on the memory-light per-token /
// no-add quant paths, where the reduce latency is on the critical path.
template <bool IS_MAX>
__device__ inline float block_reduce_1d(float v)
{
    constexpr int W = opus::get_warp_size();
    const int lane  = opus::lane_id();
    const int nwarp = opus::block_size_x() / W;
    if constexpr(W == 64)
        v = warp_reduce_dpp<IS_MAX>(v);
    else
#pragma unroll
        for(int k = W >> 1; k > 0; k >>= 1)
        {
            float o = opus::shfl(v, lane ^ k);
            v       = IS_MAX ? fmaxf(v, o) : v + o;
        }
    if(nwarp == 1)
        return v;
    __shared__ float s[64];
    const int warp = opus::thread_id_x() / W;
    opus::sync_threads(); // leading barrier: s[] is reused across the sum + max reduces
    if(lane == 0)
        s[warp] = v;
    opus::sync_threads();
    float r = (lane < nwarp) ? s[lane] : (IS_MAX ? -3.4e38f : 0.0f);
#pragma unroll
    for(int k = W >> 1; k > 0; k >>= 1)
    {
        float o = opus::shfl(r, lane ^ k);
        r       = IS_MAX ? fmaxf(r, o) : r + o;
    }
    return r; // every lane holds the full-block reduction
}

template <typename scalar_t, int width>
using vec_t = scalar_t __attribute__((ext_vector_type(width)));

template <typename Traits>
__global__ void rmsnorm_be_opus(void* __restrict__ out_,
                                        const void* __restrict__ in_,
                                        const void* __restrict__ weight_,
                                        void* __restrict__ residual_,
                                        float epsilon,
                                        int rows,
                                        int hidden,
                                        int model_sensitive)
{
    using scalar_t    = typename Traits::scalar_t;
    constexpr int TN  = Traits::TN;
    constexpr int RN  = Traits::RN;
    using V           = vec_t<scalar_t, 8>;
    const bool t5     = model_sensitive != 0;
    const bool add    = residual_ != nullptr;
    const int nx      = opus::thread_id_x(); // 0..TN-1, thread within row
    const int row     = opus::block_id_x() * opus::block_size_y() + opus::thread_id_y();
    const bool active = row < rows;
    const size_t roff = (size_t)(active ? row : 0) * hidden;

    auto* out_v      = reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(out_) + roff);
    const auto* in_v = reinterpret_cast<const V*>(reinterpret_cast<const scalar_t*>(in_) + roff);
    const auto* w_v  = reinterpret_cast<const V*>(reinterpret_cast<const scalar_t*>(weight_));
    auto* res_v      = reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(residual_) + roff);

    // fp32 norm-input as a scalar array so the compiler can't reorder the sum.
    float ni[RN][8];
#pragma unroll
    for(int q = 0; q < RN; ++q)
    {
        V x = in_v[nx + q * TN];
        if(add)
        {
            V s;
#pragma unroll
            for(int j = 0; j < 8; ++j)
            {
                float f  = opus::cast<float>(x[j]) + opus::cast<float>(res_v[nx + q * TN][j]);
                s[j]     = opus::cast<scalar_t>(f);
                ni[q][j] = t5 ? opus::cast<float>(s[j]) : f;
            }
            res_v[nx + q * TN] = s;
        }
        else
        {
#pragma unroll
            for(int j = 0; j < 8; ++j)
                ni[q][j] = opus::cast<float>(x[j]);
        }
    }

    // squared-sum in CK's order: T5 sums pairs, default one at a time.
    float sq = 0.0f;
    if(t5)
    {
#pragma unroll
        for(int q = 0; q < RN; ++q)
#pragma unroll
            for(int j = 0; j < 8; j += 2)
                sq += ni[q][j] * ni[q][j] + ni[q][j + 1] * ni[q][j + 1];
    }
    else
    {
#pragma unroll
        for(int q = 0; q < RN; ++q)
#pragma unroll
            for(int j = 0; j < 8; ++j)
                sq += ni[q][j] * ni[q][j];
    }

    // within-warp butterfly over the row's TN-lane group
    const int lane = opus::lane_id();
#pragma unroll
    for(int k = 1; k < TN && k < 64; k <<= 1)
        sq += opus::shfl(sq, lane ^ k);

    float total;
    if constexpr(TN > 64)
    {
        // cross-warp tree over W = TN/64 warps of this row (1 row per block here)
        constexpr int W = TN / 64;
        __shared__ float ws[W];
        if(lane == 0)
            ws[nx / 64] = sq;
        opus::sync_threads();
        float v[W];
#pragma unroll
        for(int i = 0; i < W; ++i)
            v[i] = ws[i];
#pragma unroll
        for(int stride = 1; stride < W; stride <<= 1)
#pragma unroll
            for(int idx = 0; idx + stride < W; idx += stride * 2)
                v[idx] += v[idx + stride];
        total = v[0];
    }
    else
        total = sq;

    if(!active)
        return;
    float inv = rsqrtf(total / hidden + epsilon);
#pragma unroll
    for(int q = 0; q < RN; ++q)
    {
        V w = w_v[nx + q * TN];
        V y;
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            float xi = ni[q][j] * inv;
            if(t5)
                xi = opus::cast<float>(opus::cast<scalar_t>(xi));
            y[j] = opus::cast<scalar_t>(xi * opus::cast<float>(w[j]));
        }
        out_v[nx + q * TN] = y;
    }
}

template <typename Traits>
__global__ void rmsnorm_opus_kernel(void* __restrict__ out_,
                                     const void* __restrict__ in_,
                                     const void* __restrict__ weight_,
                                     void* __restrict__ residual_,
                                     float epsilon,
                                     int rows,
                                     int hidden,
                                     int model_sensitive)
{
    using scalar_t        = typename Traits::scalar_t;
    constexpr int width   = Traits::width;
    constexpr bool GEMMA  = Traits::gemma;
    using V               = vec_t<scalar_t, width>;
    using Vf              = vec_t<float, width>;
    const bool t5         = model_sensitive != 0;
    const bool add       = residual_ != nullptr;
    const int lane       = opus::thread_id_x();
    const int tpr        = opus::block_size_x();
    const int row        = opus::block_id_x() * opus::block_size_y() + opus::thread_id_y();
    const int vec_hidden = hidden / width;
    const bool active    = row < rows;
    const size_t roff    = (size_t)(active ? row : 0) * hidden;

    auto* out_v      = reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(out_) + roff);
    const auto* in_v = reinterpret_cast<const V*>(reinterpret_cast<const scalar_t*>(in_) + roff);
    const auto* w_v  = reinterpret_cast<const V*>(reinterpret_cast<const scalar_t*>(weight_));
    auto* res_v      = reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(residual_) + roff);

    // fp32 norm-input cached in registers (overflow reloads).
    constexpr int CACHE_V = 4;
    Vf cache[CACHE_V];
    float acc    = 0.0f;
    auto load_ni = [&](int idx) -> Vf {
        V x = in_v[idx];
        Vf ni;
        if(add)
        {
            V s;
#pragma unroll
            for(int j = 0; j < width; ++j)
            {
                float f = opus::cast<float>(x[j]) + opus::cast<float>(res_v[idx][j]);
                s[j]    = opus::cast<scalar_t>(f);
                ni[j]   = t5 ? opus::cast<float>(s[j]) : f;
            }
            res_v[idx] = s;
        }
        else
        {
#pragma unroll
            for(int j = 0; j < width; ++j)
                ni[j] = opus::cast<float>(x[j]);
        }
        return ni;
    };
    auto reload_ni = [&](int idx) -> Vf { // overflow: residual already holds round(sum)
        V s = add ? res_v[idx] : in_v[idx];
        Vf ni;
#pragma unroll
        for(int j = 0; j < width; ++j)
            ni[j] = opus::cast<float>(s[j]);
        return ni;
    };
    auto sumsq = [&](Vf ni) {
#pragma unroll
        for(int j = 0; j < width; ++j)
            acc += ni[j] * ni[j];
    };
    if(active)
    {
#pragma unroll
        for(int k = 0; k < CACHE_V; ++k)
        {
            const int idx = lane + k * tpr;
            if(idx < vec_hidden)
            {
                cache[k] = load_ni(idx);
                sumsq(cache[k]);
            }
        }
        for(int idx = lane + CACHE_V * tpr; idx < vec_hidden; idx += tpr)
            sumsq(load_ni(idx));
    }

    float inv = rsqrtf(block_reduce<false>(acc) / hidden + epsilon);
    if(!active)
        return;

    auto store = [&](Vf ni, int idx) {
        V w = w_v[idx];
        V y;
#pragma unroll
        for(int j = 0; j < width; ++j)
        {
            float xi = ni[j] * inv;
            if(t5)
                xi = opus::cast<float>(opus::cast<scalar_t>(xi));
            // GEMMA folds (weight + 1) at compile time; GEMMA==false adds nothing.
            if constexpr(GEMMA)
                y[j] = opus::cast<scalar_t>(xi * (opus::cast<float>(w[j]) + 1.0f));
            else
                y[j] = opus::cast<scalar_t>(xi * opus::cast<float>(w[j]));
        }
        out_v[idx] = y;
    };
#pragma unroll
    for(int k = 0; k < CACHE_V; ++k)
    {
        const int idx = lane + k * tpr;
        if(idx < vec_hidden)
            store(cache[k], idx);
    }
    for(int idx = lane + CACHE_V * tpr; idx < vec_hidden; idx += tpr)
        store(reload_ni(idx), idx);
}

template <typename Traits>
__global__ void rmsnorm_quant_opus(void* __restrict__ out_,
                                       void* __restrict__ yscale_,
                                       void* __restrict__ unquant_,
                                       const void* __restrict__ in_,
                                       const void* __restrict__ weight_,
                                       void* __restrict__ residual_,
                                       const void* __restrict__ xscale_,
                                       float epsilon,
                                       int rows,
                                       int hidden,
                                       float qmax,
                                       int model_sensitive)
{
    using in_t           = typename Traits::in_t;
    using out_t          = typename Traits::out_t;
    constexpr int width  = Traits::width;
    using Vi             = vec_t<in_t, width>;
    using Vo             = vec_t<out_t, width>;
    using Vf             = vec_t<float, width>;
    const bool add       = residual_ != nullptr;
    const bool smooth    = xscale_ != nullptr;
    const bool save_uq   = unquant_ != nullptr;
    const bool t5        = model_sensitive != 0;
    const int lane       = opus::thread_id_x();
    const int tpr        = opus::block_size_x();
    const int row        = opus::block_id_x() * opus::block_size_y() + opus::thread_id_y();
    const int vec_hidden = hidden / width;
    const bool active    = row < rows;
    const size_t roff    = (size_t)(active ? row : 0) * hidden;

    const auto* in_v   = reinterpret_cast<const Vi*>(reinterpret_cast<const in_t*>(in_) + roff);
    const auto* w_v    = reinterpret_cast<const Vi*>(reinterpret_cast<const in_t*>(weight_));
    auto* out_v        = reinterpret_cast<Vo*>(reinterpret_cast<out_t*>(out_) + roff);
    auto* res_v        = reinterpret_cast<Vi*>(reinterpret_cast<in_t*>(residual_) + roff);
    auto* uq_v         = reinterpret_cast<Vi*>(reinterpret_cast<in_t*>(unquant_) + roff);
    const auto* xscale = reinterpret_cast<const float*>(xscale_);

    // fp32 norm-input cached (see the norm kernel).
    constexpr int CACHE_V = 4;
    Vf cache[CACHE_V];
    float acc    = 0.0f;
    auto load_ni = [&](int idx) -> Vf {
        Vi x = in_v[idx];
        Vf ni;
        if(add)
        {
            Vi s;
#pragma unroll
            for(int j = 0; j < width; ++j)
            {
                float f = opus::cast<float>(x[j]) + opus::cast<float>(res_v[idx][j]);
                s[j]    = opus::cast<in_t>(f);
                ni[j]   = t5 ? opus::cast<float>(s[j]) : f;
            }
            res_v[idx] = s;
        }
        else
        {
#pragma unroll
            for(int j = 0; j < width; ++j)
                ni[j] = opus::cast<float>(x[j]);
        }
        return ni;
    };
    auto reload_ni = [&](int idx) -> Vf {
        Vi s = add ? res_v[idx] : in_v[idx];
        Vf ni;
#pragma unroll
        for(int j = 0; j < width; ++j)
            ni[j] = opus::cast<float>(s[j]);
        return ni;
    };
    auto sumsq = [&](Vf ni) {
#pragma unroll
        for(int j = 0; j < width; ++j)
            acc += ni[j] * ni[j];
    };
    if(active)
    {
#pragma unroll
        for(int k = 0; k < CACHE_V; ++k)
        {
            const int idx = lane + k * tpr;
            if(idx < vec_hidden)
            {
                cache[k] = load_ni(idx);
                sumsq(cache[k]);
            }
        }
        for(int idx = lane + CACHE_V * tpr; idx < vec_hidden; idx += tpr)
            sumsq(load_ni(idx));
    }

    float inv = rsqrtf(block_reduce<false>(acc) / hidden + epsilon);

    // normalized value: n = (t5 ? round(ni*inv) : ni*inv) * w [* xscale]
    auto norm_j = [&](float ni, in_t wval, int col) -> float {
        float xi = ni * inv;
        if(t5)
            xi = opus::cast<float>(opus::cast<in_t>(xi));
        float n = xi * opus::cast<float>(wval);
        return smooth ? n * xscale[col] : n;
    };

    float m = 0.0f;
    if(active)
    {
        auto absmax = [&](Vf ni, int idx) {
            Vi w = w_v[idx];
#pragma unroll
            for(int j = 0; j < width; ++j)
                m = fmaxf(m, fabsf(norm_j(ni[j], w[j], idx * width + j)));
        };
#pragma unroll
        for(int k = 0; k < CACHE_V; ++k)
        {
            const int idx = lane + k * tpr;
            if(idx < vec_hidden)
                absmax(cache[k], idx);
        }
        for(int idx = lane + CACHE_V * tpr; idx < vec_hidden; idx += tpr)
            absmax(reload_ni(idx), idx);
    }

    float rowmax = block_reduce<true>(m);
    if(!active)
        return;
    float yscale = rowmax / qmax;
    float inv_ys = yscale > 0.0f ? 1.0f / yscale : 0.0f;
    if(lane == 0)
        reinterpret_cast<float*>(yscale_)[row] = yscale;

    auto quant = [&](Vf ni, int idx) {
        Vi w = w_v[idx];
        Vo q;
        Vi uq;
#pragma unroll
        for(int j = 0; j < width; ++j)
        {
            float n = norm_j(ni[j], w[j], idx * width + j);
            q[j]    = quant_cast<out_t>(n * inv_ys);
            if(save_uq)
                uq[j] = opus::cast<in_t>(n);
        }
        out_v[idx] = q;
        if(save_uq)
            uq_v[idx] = uq;
    };
#pragma unroll
    for(int k = 0; k < CACHE_V; ++k)
    {
        const int idx = lane + k * tpr;
        if(idx < vec_hidden)
            quant(cache[k], idx);
    }
    for(int idx = lane + CACHE_V * tpr; idx < vec_hidden; idx += tpr)
        quant(reload_ni(idx), idx);
}

template <typename Traits>
__global__ void add_rmsnorm_quant_opus(void* __restrict__ out_,
                                       void* __restrict__ rout_,
                                       void* __restrict__ scale_,
                                       const void* __restrict__ in_,
                                       const void* __restrict__ rin_,
                                       const void* __restrict__ w_,
                                       const void* __restrict__ xscale_,
                                       float epsilon,
                                       int m,
                                       int n,
                                       float qmax,
                                       int in_s,
                                       int rin_s,
                                       int rout_s,
                                       int out_s,
                                       int group_size,
                                       int shuffle,
                                       int gemma)
{
    using in_t         = typename Traits::in_t;
    using out_t        = typename Traits::out_t;
    constexpr int BLK  = Traits::BLK;
    constexpr int TDS  = Traits::TDS;
    constexpr bool ILV = Traits::ILV;
    constexpr bool FP4 = std::is_same_v<out_t, opus::fp4_t>;
    constexpr bool QNT = !std::is_same_v<out_t, in_t>; // quant when out dtype != in
    const bool ADD     = rin_ != nullptr;              // fused-add when residual given
    (void)xscale_;                                     // smooth quant lives in module_rmsnorm
    const int row      = opus::block_id_x();
    if(row >= m)
        return;
    const int tid    = opus::thread_id_x();
    const float goff = gemma ? 1.0f : 0.0f;

    // Vectorized coalesced IO, mirroring add_rmsnorm_quant_kernel (opus_vec_io layer).
    using out_store_t = std::conditional_t<FP4, uint8_t, out_t>;
    constexpr int load_chunk_bytes = (sizeof(in_t) * TDS % 16 == 0) ? 16 : 8;
    constexpr int load_vec_size    = load_chunk_bytes / sizeof(in_t);
    constexpr int num_load_inst    = TDS / load_vec_size;
    constexpr int load_aux         = (num_load_inst > 1 && !ILV) ? RT : GROUP_NT;
    constexpr int ISZ              = opus::get_warp_size(); // interleave_thread_size
    constexpr int ooba_i           = 4 / sizeof(in_t);
    const int oob_i                = (n + ooba_i - 1) / ooba_i * ooba_i;

    // Interleaved (per-token) layout permutes each thread's columns for coalescing;
    // contiguous (grouped) layout keeps tid's TDS columns adjacent for group reduction.
    const int row_offset = (ILV && num_load_inst > 1)
                               ? (tid % ISZ * load_vec_size + tid / ISZ * ISZ * TDS)
                               : (tid * TDS);
    const int col0 = tid * TDS; // logical first column (grouped layout / guards)

    auto buf_i = opus::make_gmem<in_t>(reinterpret_cast<const in_t*>(in_) + (size_t)row * in_s,
                                       oob_i * sizeof(in_t));
    auto buf_w = opus::make_gmem<in_t>(reinterpret_cast<const in_t*>(w_), oob_i * sizeof(in_t));
    auto td_i  = load_vector_nbytes<in_t, TDS, load_chunk_bytes, load_aux, ILV, ISZ>(buf_i, row_offset);
    auto td_w  = load_vector_nbytes<in_t, TDS, load_chunk_bytes, RT, ILV, ISZ>(buf_w, row_offset);

    opus::vector_t<float, TDS> f; // fp32 norm-input; residual stored as round(x+res)
    if(ADD)
    {
        auto buf_r = opus::make_gmem<in_t>(reinterpret_cast<const in_t*>(rin_) + (size_t)row * rin_s,
                                           oob_i * sizeof(in_t));
        auto td_r =
            load_vector_nbytes<in_t, TDS, load_chunk_bytes, load_aux, ILV, ISZ>(buf_r, row_offset);
#pragma unroll
        for(int j = 0; j < TDS; ++j)
            f[j] = opus::cast<float>(td_i[j]) + opus::cast<float>(td_r[j]);
        auto buf_ro = opus::make_gmem<in_t>(reinterpret_cast<in_t*>(rout_) + (size_t)row * rout_s,
                                            oob_i * sizeof(in_t));
        store_vector<in_t, float, TDS, load_aux, ILV, ISZ, num_load_inst, in_t>(buf_ro, f,
                                                                                row_offset);
    }
    else
    {
#pragma unroll
        for(int j = 0; j < TDS; ++j)
            f[j] = opus::cast<float>(td_i[j]);
    }

    // packed square-sum + normalize (v_pk_mul_f32 on CDNA), matching the reference.
    auto* f2 = reinterpret_cast<opus::fp32x2_t*>(&f);
    float sq = 0.0f;
#pragma unroll
    for(int j = 0; j < TDS / 2; ++j)
    {
        opus::fp32x2_t p = pk_mul_f32(f2[j], f2[j]);
        sq += p[0] + p[1];
    }
    float inv = rsqrtf(block_reduce_1d<false>(sq) / n + epsilon);

    const opus::fp32x2_t invv{inv, inv};
#pragma unroll
    for(int j = 0; j < TDS / 2; ++j) // normalized: (f*inv) * (w + goff)
    {
        opus::fp32x2_t wv{opus::cast<float>(td_w[2 * j]) + goff,
                          opus::cast<float>(td_w[2 * j + 1]) + goff};
        f2[j] = pk_mul_f32(pk_mul_f32(f2[j], invv), wv);
    }

    constexpr int ooba_o = 4 / sizeof(out_store_t);
    const int oob_o      = (n + ooba_o - 1) / ooba_o * ooba_o;

    if constexpr(!QNT)
    {
        auto buf_o = opus::make_gmem<out_store_t>(
            reinterpret_cast<out_store_t*>(out_) + (size_t)row * out_s, oob_o * sizeof(out_store_t));
        store_vector<out_store_t, float, TDS, RT, ILV, ISZ, num_load_inst, out_t>(buf_o, f,
                                                                                  row_offset);
        return;
    }

    float tmax = 1e-10f;
#pragma unroll
    for(int j = 0; j < TDS; ++j)
        tmax = fmaxf(tmax, fabsf(f[j]));

    const float qm = FP4 ? 6.0f : qmax;
    float inv_ys; // store scale factor passed to store_vector (fp4: forward e8m0)
    if(group_size == 0)
    {
        float gmax  = block_reduce_1d<true>(tmax);
        float ys    = gmax / qm;
        inv_ys      = ys > 0.0f ? 1.0f / ys : 0.0f;
        if(tid == 0 && scale_)
            reinterpret_cast<float*>(scale_)[row] = ys;
    }
    else
    {
        const int rts  = group_size / TDS; // contiguous lanes per group
        const int lane = opus::lane_id();
#pragma unroll
        for(int k = 1; k < 64; k <<= 1)
            if(k < rts)
                tmax = fmaxf(tmax, opus::shfl(tmax, lane ^ k));
        float ys;
        if constexpr(FP4)
        {
            ys     = fp4_e8m0_scale(tmax);
            inv_ys = ys; // opus fp4 packer takes the forward scale
        }
        else
        {
            ys     = tmax / qm;
            inv_ys = ys > 0.0f ? 1.0f / ys : 0.0f;
        }
        if((tid % rts) == 0 && col0 < n && scale_)
        {
            int y      = tid / rts;
            int groups = n / group_size;
            if constexpr(FP4)
            {
                int sp    = shuffle ? (groups + 7) / 8 * 8 : groups;
                size_t si = shuffle ? (size_t)mx_scale_shuffle_idx(sp, row, y) : (size_t)row * sp + y;
                reinterpret_cast<unsigned char*>(scale_)[si] = e8m0_byte(ys);
            }
            else
            {
                size_t si = shuffle ? (size_t)y * m + row : (size_t)row * groups + y;
                reinterpret_cast<float*>(scale_)[si] = ys;
            }
        }
    }

    const int store_off = FP4 ? row_offset / 2 : row_offset;
    auto buf_o          = opus::make_gmem<out_store_t>(
        reinterpret_cast<out_store_t*>(out_) + (size_t)row * out_s, oob_o * sizeof(out_store_t));
    store_vector<out_store_t, float, TDS, RT, ILV, ISZ, num_load_inst, out_t>(buf_o, f, store_off,
                                                                              inv_ys);
}

#endif // __HIP_DEVICE_COMPILE__

} // namespace aiter
