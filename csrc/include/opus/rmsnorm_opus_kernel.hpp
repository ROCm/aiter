// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm device kernels. Host/device split: opus.hpp is parsed only on the
// device pass; the host pass sees declarations + empty stubs. 2D block:
// blockDim.x = threads-per-row, blockDim.y = rows-per-block.
#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"
#endif

namespace aiter {
namespace rmsnorm_opus {

// Element types matching opus REGISTER_DTYPE so both passes name the same type.
#if defined(__clang_major__) && __clang_major__ >= 20
using bf16_t = __bf16;
using fp16_t = __fp16;
#else
using bf16_t = unsigned short;
using fp16_t = _Float16;
#endif
using i8_t  = signed char;
using fp8_t = _BitInt(8);

// rmsnorm, optionally fused with a residual add. residual != 0: s = in + residual,
// residual = s (pre-norm), out = rmsnorm(s) * weight (in-place when out == in).
// model_sensitive != 0 selects the T5 variant (round s*inv to dtype before *w).
template <typename scalar_t, int width>
__global__ void rmsnorm2d_fwd_kernel(void* __restrict__ out,
                                     const void* __restrict__ in,
                                     const void* __restrict__ weight,
                                     void* __restrict__ residual,
                                     float epsilon,
                                     int rows,
                                     int hidden,
                                     int model_sensitive);

// rmsnorm + dynamic/smooth quant. Flags via pointers: residual != 0 fused-add,
// xscale != 0 smooth (per-col premultiply), unquant != 0 store pre-quant y.
// out is int8/fp8; yscale is [rows] fp32 (rowmax/qmax).
template <typename in_t, typename out_t, int width>
__global__ void rmsnorm2d_quant_kernel(void* __restrict__ out,
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

// Bit-exact (vs CK) rmsnorm, optional fused-add. Replicates CK's reduction order
// for CK's tile geometry: TN threads/row own RN vectors (width 8) each; sum-of-
// squares is a paired intra-thread sum + butterfly xor within the warp + a
// cross-warp tree over TN/64 warps. Reproduces CK's square_sum bit-for-bit.
template <typename scalar_t, int TN, int RN>
__global__ void rmsnorm2d_fwd_be_kernel(void* __restrict__ out,
                                        const void* __restrict__ in,
                                        const void* __restrict__ weight,
                                        void* __restrict__ residual,
                                        float epsilon,
                                        int rows,
                                        int hidden,
                                        int model_sensitive);

#if !defined(__HIP_DEVICE_COMPILE__)
// Host pass: empty stubs so the __device_stub__ symbols resolve.
template <typename scalar_t, int width>
__global__ void rmsnorm2d_fwd_kernel(void*, const void*, const void*, void*, float, int, int, int)
{
}
template <typename in_t, typename out_t, int width>
__global__ void rmsnorm2d_quant_kernel(
    void*, void*, void*, const void*, const void*, void*, const void*, float, int, int, float, int)
{
}
template <typename scalar_t, int TN, int RN>
__global__ void
rmsnorm2d_fwd_be_kernel(void*, const void*, const void*, void*, float, int, int, int)
{
}
#else
// bf16/fp16 are native types; f32 conversions are exact widening / round-to-nearest.
template <typename scalar_t>
__device__ inline float to_f32(scalar_t x)
{ return static_cast<float>(x); }
template <typename scalar_t>
__device__ inline scalar_t from_f32(float x)
{ return static_cast<scalar_t>(x); }

// fp32 -> quant element. int8: round-to-nearest; fp8: hardware e4m3 cvt.
template <typename out_t>
__device__ inline out_t quant_cast(float v)
{
    if constexpr(std::is_same_v<out_t, i8_t>)
        return static_cast<i8_t>(__builtin_rintf(v));
    else
        return opus::fp32_to_fp8(v);
}

// Per-row (segmented) sequential-addressing LDS reduction over blockDim.x threads;
// bank-conflict-free and deterministic (all rows step the same stride sequence).
template <bool IS_MAX>
__device__ inline float block_reduce(float v)
{
    __shared__ float s[1024];
    const int lane = opus::thread_id_x();
    const int tpr  = opus::block_size_x();
    const int base = opus::thread_id_y() * tpr;
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

template <typename scalar_t, int TN, int RN>
__global__ void rmsnorm2d_fwd_be_kernel(void* __restrict__ out_,
                                        const void* __restrict__ in_,
                                        const void* __restrict__ weight_,
                                        void* __restrict__ residual_,
                                        float epsilon,
                                        int rows,
                                        int hidden,
                                        int model_sensitive)
{
    using V           = vec_t<scalar_t, 8>;
    using Vf          = vec_t<float, 8>;
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

    // norm-input in fp32; fused-add writes round(x+res) to residual, keeps fp32
    // sum (default) or rounded sum (T5) for the norm.
    Vf ni[RN];
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
                float f  = to_f32<scalar_t>(x[j]) + to_f32<scalar_t>(res_v[nx + q * TN][j]);
                s[j]     = from_f32<scalar_t>(f);
                ni[q][j] = t5 ? to_f32<scalar_t>(s[j]) : f;
            }
            res_v[nx + q * TN] = s;
        }
        else
        {
#pragma unroll
            for(int j = 0; j < 8; ++j)
                ni[q][j] = to_f32<scalar_t>(x[j]);
        }
    }

    // intra-thread squared-sum in CK's order: T5 pipeline sums in pairs
    // (a0^2+a1^2), the default pipeline sums one element at a time.
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
                xi = to_f32<scalar_t>(from_f32<scalar_t>(xi));
            y[j] = from_f32<scalar_t>(xi * to_f32<scalar_t>(w[j]));
        }
        out_v[nx + q * TN] = y;
    }
}

template <typename scalar_t, int width>
__global__ void rmsnorm2d_fwd_kernel(void* __restrict__ out_,
                                     const void* __restrict__ in_,
                                     const void* __restrict__ weight_,
                                     void* __restrict__ residual_,
                                     float epsilon,
                                     int rows,
                                     int hidden,
                                     int model_sensitive)
{
    using V              = vec_t<scalar_t, width>;
    using Vf             = vec_t<float, width>;
    const bool t5        = model_sensitive != 0;
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

    // fp32 norm-input, cached in registers (overflow reloads). Fused-add writes
    // round(x+res) to residual, but the norm keeps the fp32 sum (default) or the
    // rounded sum (T5, matching vLLM). Non-fused: just f32(in).
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
                float f = to_f32<scalar_t>(x[j]) + to_f32<scalar_t>(res_v[idx][j]);
                s[j]    = from_f32<scalar_t>(f);
                ni[j]   = t5 ? to_f32<scalar_t>(s[j]) : f;
            }
            res_v[idx] = s;
        }
        else
        {
#pragma unroll
            for(int j = 0; j < width; ++j)
                ni[j] = to_f32<scalar_t>(x[j]);
        }
        return ni;
    };
    auto reload_ni = [&](int idx) -> Vf { // overflow: residual already holds round(sum)
        V s = add ? res_v[idx] : in_v[idx];
        Vf ni;
#pragma unroll
        for(int j = 0; j < width; ++j)
            ni[j] = to_f32<scalar_t>(s[j]);
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
                xi = to_f32<scalar_t>(from_f32<scalar_t>(xi));
            y[j] = from_f32<scalar_t>(xi * to_f32<scalar_t>(w[j]));
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

template <typename in_t, typename out_t, int width>
__global__ void rmsnorm2d_quant_kernel(void* __restrict__ out_,
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

    // fp32 norm-input, cached (see the norm kernel). Fused-add writes round(x+res)
    // to residual; the norm keeps the fp32 sum (default) or the rounded sum (T5).
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
                float f = to_f32<in_t>(x[j]) + to_f32<in_t>(res_v[idx][j]);
                s[j]    = from_f32<in_t>(f);
                ni[j]   = t5 ? to_f32<in_t>(s[j]) : f;
            }
            res_v[idx] = s;
        }
        else
        {
#pragma unroll
            for(int j = 0; j < width; ++j)
                ni[j] = to_f32<in_t>(x[j]);
        }
        return ni;
    };
    auto reload_ni = [&](int idx) -> Vf {
        Vi s = add ? res_v[idx] : in_v[idx];
        Vf ni;
#pragma unroll
        for(int j = 0; j < width; ++j)
            ni[j] = to_f32<in_t>(s[j]);
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
            xi = to_f32<in_t>(from_f32<in_t>(xi));
        float n = xi * to_f32<in_t>(wval);
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
                uq[j] = from_f32<in_t>(n);
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

} // namespace rmsnorm_opus
} // namespace aiter
