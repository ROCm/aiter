// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm device kernels. 2D block: x = threads/row, y = rows/block.
#pragma once

// fp32->bf16 store rounding (must precede opus.hpp). Use truncate (2), matching the CK/HIP
// reference: RNE (0) has no hardware bf16-cvt on gfx942 and lowers to a ~6-op/element
// software sequence, making bf16 output/residual stores ~2x slower there (gfx950 has the
// hardware cvt so RNE is free, but truncate matches the reference on both).
#ifndef OPUS_FP32_to_BF16_DEFAULT
#define OPUS_FP32_to_BF16_DEFAULT 2
#endif
#include "opus/opus.hpp"

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
template <typename Traits, bool OOP>
__global__ void rmsnorm_opus_kernel(void* __restrict__ out,
                                     const void* __restrict__ in,
                                     const void* __restrict__ weight,
                                     void* __restrict__ residual,
                                     void* __restrict__ residual_out,
                                     float epsilon,
                                     int rows,
                                     int hidden,
                                     int in_s,
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
template <typename Traits, bool OOP>
__global__ void rmsnorm_be_opus(void* __restrict__ out,
                                        const void* __restrict__ in,
                                        const void* __restrict__ weight,
                                        void* __restrict__ residual,
                                        void* __restrict__ residual_out,
                                        float epsilon,
                                        int rows,
                                        int hidden,
                                        int in_s,
                                        int model_sensitive);

#if !defined(__HIP_DEVICE_COMPILE__)
// Host pass: empty stubs so the __device_stub__ symbols resolve.
template <typename Traits, bool OOP>
__global__ void rmsnorm_opus_kernel(void*, const void*, const void*, void*, void*, float, int, int, int, int)
{
}
template <typename Traits>
__global__ void rmsnorm_quant_opus(
    void*, void*, void*, const void*, const void*, void*, const void*, float, int, int, float, int)
{
}
template <typename Traits, bool OOP>
__global__ void
rmsnorm_be_opus(void*, const void*, const void*, void*, void*, float, int, int, int, int)
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

template <typename scalar_t, int width>
using vec_t = scalar_t __attribute__((ext_vector_type(width)));

// OOP=false: in-place fused add (residual_out_ unused). OOP=true: out-of-place.
template <typename Traits, bool OOP>
__global__ void rmsnorm_be_opus(void* __restrict__ out_,
                                        const void* __restrict__ in_,
                                        const void* __restrict__ weight_,
                                        void* __restrict__ residual_,
                                        void* __restrict__ residual_out_,
                                        float epsilon,
                                        int rows,
                                        int hidden,
                                        int in_s,
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
    // out/residual are contiguous (row*hidden); input may be row-strided (row*in_s).
    const size_t roff   = (size_t)(active ? row : 0) * hidden;
    const size_t roff_i = (size_t)(active ? row : 0) * in_s;

    auto* out_v      = reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(out_) + roff);
    const auto* in_v = reinterpret_cast<const V*>(reinterpret_cast<const scalar_t*>(in_) + roff_i);
    const auto* w_v  = reinterpret_cast<const V*>(reinterpret_cast<const scalar_t*>(weight_));
    auto* res_v      = reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(residual_) + roff);
    auto* res_out_v  = OOP ? reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(residual_out_) + roff)
                           : static_cast<V*>(nullptr);

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
            if constexpr(OOP)
                res_out_v[nx + q * TN] = s;
            else
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

// OOP=false: in-place fused add (residual read+written in residual_; residual_out_ unused
// so the no-add / in-place instantiation compiles identically to the pre-out-of-place
// kernel). OOP=true: out-of-place fused add (read residual_, write residual_out_).
template <typename Traits, bool OOP>
__global__ void rmsnorm_opus_kernel(void* __restrict__ out_,
                                     const void* __restrict__ in_,
                                     const void* __restrict__ weight_,
                                     void* __restrict__ residual_,
                                     void* __restrict__ residual_out_,
                                     float epsilon,
                                     int rows,
                                     int hidden,
                                     int in_s,
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
    // out/residual are contiguous (row*hidden); input may be row-strided (row*in_s).
    const size_t roff    = (size_t)(active ? row : 0) * hidden;
    const size_t roff_i  = (size_t)(active ? row : 0) * in_s;

    auto* out_v      = reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(out_) + roff);
    const auto* in_v = reinterpret_cast<const V*>(reinterpret_cast<const scalar_t*>(in_) + roff_i);
    const auto* w_v  = reinterpret_cast<const V*>(reinterpret_cast<const scalar_t*>(weight_));
    auto* res_v      = reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(residual_) + roff);
    // res_out_v is dead (nullptr) for OOP=false, so it costs the in-place/no-add path nothing.
    auto* res_out_v  = OOP ? reinterpret_cast<V*>(reinterpret_cast<scalar_t*>(residual_out_) + roff)
                           : static_cast<V*>(nullptr);

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
            if constexpr(OOP)
                res_out_v[idx] = s;
            else
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
    auto reload_ni = [&](int idx) -> Vf { // overflow: residual (out) already holds round(sum)
        V s = add ? (OOP ? res_out_v[idx] : res_v[idx]) : in_v[idx];
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

#endif // __HIP_DEVICE_COMPILE__

} // namespace aiter
