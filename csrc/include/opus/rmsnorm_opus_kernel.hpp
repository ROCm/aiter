// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm device kernels. Host/device pass split: opus.hpp is parsed only
// on the device pass; the host pass sees declarations + empty stubs.
#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"
#endif

namespace aiter {
namespace rmsnorm_opus {

// Element types matching opus's REGISTER_DTYPE so both passes name the same type.
#if defined(__clang_major__) && __clang_major__ >= 20
using bf16_t = __bf16;
using fp16_t = __fp16;
#else
using bf16_t = unsigned short;
using fp16_t = _Float16;
#endif
// quant output element types (match opus REGISTER_DTYPE i8/fp8)
using i8_t  = signed char;
using fp8_t = _BitInt(8);

// out[t,i] = scalar( f32(in[t,i]) * rsqrt(mean_i(in[t,:]^2) + eps) * f32(w[i]) )
// model_sensitive != 0 selects the T5 variant (round s*inv to dtype before *w).
template <typename scalar_t, int width>
__global__ void rmsnorm2d_fwd_kernel(void* __restrict__ out,
                                     const void* __restrict__ in,
                                     const void* __restrict__ weight,
                                     float epsilon,
                                     int rows,
                                     int hidden,
                                     int model_sensitive);

// Fused residual add + rmsnorm, in place:
//   x = in + residual;  residual = x;  in = rmsnorm(x) * weight
template <typename scalar_t, int width>
__global__ void fused_add_rmsnorm2d_fwd_kernel(void* __restrict__ inout,
                                               void* __restrict__ residual,
                                               const void* __restrict__ weight,
                                               float epsilon,
                                               int rows,
                                               int hidden,
                                               int model_sensitive);

// Fused rmsnorm + dynamic/smooth quant. Runtime flags via pointers: residual!=0
// => fused-add; xscale!=0 => smooth (per-col premultiply); unquant!=0 => also
// store pre-quant y. out is int8/fp8; yscale is [rows] fp32 (rowmax/qmax).
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

#if !defined(__HIP_DEVICE_COMPILE__)
// Host pass: empty stubs so the __device_stub__ symbols resolve.
template <typename scalar_t, int width>
__global__ void rmsnorm2d_fwd_kernel(void*, const void*, const void*, float, int, int, int)
{
}
template <typename scalar_t, int width>
__global__ void fused_add_rmsnorm2d_fwd_kernel(void*, void*, const void*, float, int, int, int)
{
}
template <typename in_t, typename out_t, int width>
__global__ void rmsnorm2d_quant_kernel(
    void*, void*, void*, const void*, const void*, void*, const void*, float, int, int, float, int)
{
}
#else
// Device pass. bf16/fp16 are native types; implicit f32 conversions are exact
// widening / round-to-nearest, matching the CK / vLLM reference.
template <typename scalar_t>
__device__ inline float to_f32(scalar_t x)
{ return static_cast<float>(x); }

template <typename scalar_t>
__device__ inline scalar_t from_f32(float x)
{ return static_cast<scalar_t>(x); }

// normalized element: (t5 ? round_to_dtype(s*inv) : s*inv) * w
template <typename scalar_t>
__device__ inline scalar_t norm_elem(scalar_t s, float inv, scalar_t w, bool t5)
{
    float xi = to_f32<scalar_t>(s) * inv;
    if(t5)
        xi = to_f32<scalar_t>(from_f32<scalar_t>(xi));
    return from_f32<scalar_t>(xi * to_f32<scalar_t>(w));
}

// Sequential-addressing LDS reductions (bank-conflict-free, deterministic).
// blockDim is a power of two (guaranteed by pick_block); idle threads pass in
// the identity so the full block participates.
__device__ inline float block_reduce_sum(float v)
{
    __shared__ float s[1024];
    const int tid = opus::thread_id_x();
    const int n   = opus::block_size_x();
    s[tid]        = v;
    opus::sync_threads();
    for(int stride = n >> 1; stride > 0; stride >>= 1)
    {
        if(tid < stride)
            s[tid] += s[tid + stride];
        opus::sync_threads();
    }
    return s[0];
}

// Block max-reduction (v >= 0 here).
__device__ inline float block_reduce_max(float v)
{
    __shared__ float s[1024];
    const int tid = opus::thread_id_x();
    const int n   = opus::block_size_x();
    s[tid]        = v;
    opus::sync_threads();
    for(int stride = n >> 1; stride > 0; stride >>= 1)
    {
        if(tid < stride)
            s[tid] = fmaxf(s[tid], s[tid + stride]);
        opus::sync_threads();
    }
    return s[0];
}

// fp32 -> quant element. int8: round-to-nearest; fp8: hardware e4m3 cvt.
template <typename out_t>
__device__ inline out_t quant_cast(float v)
{
    if constexpr(std::is_same_v<out_t, i8_t>)
        return static_cast<i8_t>(__builtin_rintf(v));
    else
        return opus::fp32_to_fp8(v);
}

// Vectorized load type: `width` elements of scalar_t as one aligned access.
template <typename scalar_t, int width>
using vec_t = scalar_t __attribute__((ext_vector_type(width)));

template <typename scalar_t, int width>
__global__ void rmsnorm2d_fwd_kernel(void* __restrict__ out_,
                                     const void* __restrict__ in_,
                                     const void* __restrict__ weight_,
                                     float epsilon,
                                     int rows,
                                     int hidden,
                                     int model_sensitive)
{
    const bool t5        = model_sensitive != 0;
    using V              = vec_t<scalar_t, width>;
    const int row        = opus::block_id_x();
    const int tid        = opus::thread_id_x();
    const int nthreads   = opus::block_size_x();
    const int vec_hidden = hidden / width;

    auto* out          = reinterpret_cast<scalar_t*>(out_);
    const auto* in     = reinterpret_cast<const scalar_t*>(in_);
    const auto* weight = reinterpret_cast<const scalar_t*>(weight_);
    const auto* in_v   = reinterpret_cast<const V*>(in + (size_t)row * hidden);
    auto* out_v        = reinterpret_cast<V*>(out + (size_t)row * hidden);
    const auto* w_v    = reinterpret_cast<const V*>(weight);

    // Cache the row in registers so the normalize pass does not re-read input
    // from global (single pass); overflow beyond the cache reloads. CACHE_V is
    // sized so typical hidden (<= CACHE_V*width*blockDim) stays fully cached.
    constexpr int CACHE_V = 4;
    V cache[CACHE_V];
    float acc = 0.0f;
#pragma unroll
    for(int k = 0; k < CACHE_V; ++k)
    {
        const int idx = tid + k * nthreads;
        if(idx < vec_hidden)
        {
            cache[k] = in_v[idx];
#pragma unroll
            for(int j = 0; j < width; ++j)
            {
                float f = to_f32<scalar_t>(cache[k][j]);
                acc += f * f;
            }
        }
    }
    for(int idx = tid + CACHE_V * nthreads; idx < vec_hidden; idx += nthreads)
    {
        V x = in_v[idx];
#pragma unroll
        for(int j = 0; j < width; ++j)
        {
            float f = to_f32<scalar_t>(x[j]);
            acc += f * f;
        }
    }

    float var = block_reduce_sum(acc);
    float inv = rsqrtf(var / hidden + epsilon);

#pragma unroll
    for(int k = 0; k < CACHE_V; ++k)
    {
        const int idx = tid + k * nthreads;
        if(idx < vec_hidden)
        {
            V w = w_v[idx];
            V y;
#pragma unroll
            for(int j = 0; j < width; ++j)
                y[j] = norm_elem<scalar_t>(cache[k][j], inv, w[j], t5);
            out_v[idx] = y;
        }
    }
    for(int idx = tid + CACHE_V * nthreads; idx < vec_hidden; idx += nthreads)
    {
        V x = in_v[idx];
        V w = w_v[idx];
        V y;
#pragma unroll
        for(int j = 0; j < width; ++j)
            y[j] = norm_elem<scalar_t>(x[j], inv, w[j], t5);
        out_v[idx] = y;
    }
}

template <typename scalar_t, int width>
__global__ void fused_add_rmsnorm2d_fwd_kernel(void* __restrict__ inout_,
                                               void* __restrict__ residual_,
                                               const void* __restrict__ weight_,
                                               float epsilon,
                                               int rows,
                                               int hidden,
                                               int model_sensitive)
{
    const bool t5        = model_sensitive != 0;
    using V              = vec_t<scalar_t, width>;
    const int row        = opus::block_id_x();
    const int tid        = opus::thread_id_x();
    const int nthreads   = opus::block_size_x();
    const int vec_hidden = hidden / width;

    auto* inout        = reinterpret_cast<scalar_t*>(inout_);
    auto* residual     = reinterpret_cast<scalar_t*>(residual_);
    const auto* weight = reinterpret_cast<const scalar_t*>(weight_);
    auto* io_v         = reinterpret_cast<V*>(inout + (size_t)row * hidden);
    auto* res_v        = reinterpret_cast<V*>(residual + (size_t)row * hidden);
    const auto* w_v    = reinterpret_cast<const V*>(weight);

    // Cache the pre-norm sum in registers (still written back to residual) so the
    // normalize pass does not re-read it from global; overflow reloads.
    constexpr int CACHE_V = 4;
    V cache[CACHE_V];
    float acc   = 0.0f;
    auto add_sq = [&](V x, V r) {
        V s;
#pragma unroll
        for(int j = 0; j < width; ++j)
        {
            float f = to_f32<scalar_t>(x[j]) + to_f32<scalar_t>(r[j]);
            s[j]    = from_f32<scalar_t>(f);
            acc += f * f;
        }
        return s;
    };
#pragma unroll
    for(int k = 0; k < CACHE_V; ++k)
    {
        const int idx = tid + k * nthreads;
        if(idx < vec_hidden)
        {
            cache[k]   = add_sq(io_v[idx], res_v[idx]);
            res_v[idx] = cache[k]; // pre-norm residual write-back
        }
    }
    for(int idx = tid + CACHE_V * nthreads; idx < vec_hidden; idx += nthreads)
        res_v[idx] = add_sq(io_v[idx], res_v[idx]);

    float var = block_reduce_sum(acc);
    float inv = rsqrtf(var / hidden + epsilon);

    auto normalize = [&](V s, V w) {
        V y;
#pragma unroll
        for(int j = 0; j < width; ++j)
            y[j] = norm_elem<scalar_t>(s[j], inv, w[j], t5);
        return y;
    };
#pragma unroll
    for(int k = 0; k < CACHE_V; ++k)
    {
        const int idx = tid + k * nthreads;
        if(idx < vec_hidden)
            io_v[idx] = normalize(cache[k], w_v[idx]);
    }
    for(int idx = tid + CACHE_V * nthreads; idx < vec_hidden; idx += nthreads)
        io_v[idx] = normalize(res_v[idx], w_v[idx]);
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
    const int row        = opus::block_id_x();
    const int tid        = opus::thread_id_x();
    const int nthreads   = opus::block_size_x();
    const int vec_hidden = hidden / width;

    const bool fused_add = residual_ != nullptr;
    const bool smooth    = xscale_ != nullptr;
    const bool save_uq   = unquant_ != nullptr;
    const bool t5        = model_sensitive != 0;

    const auto* in     = reinterpret_cast<const in_t*>(in_);
    const auto* weight = reinterpret_cast<const in_t*>(weight_);
    const auto* in_v   = reinterpret_cast<const Vi*>(in + (size_t)row * hidden);
    const auto* w_v    = reinterpret_cast<const Vi*>(weight);
    auto* out_v = reinterpret_cast<Vo*>(reinterpret_cast<out_t*>(out_) + (size_t)row * hidden);
    auto* res_v = reinterpret_cast<Vi*>(reinterpret_cast<in_t*>(residual_) + (size_t)row * hidden);
    auto* uq_v  = reinterpret_cast<Vi*>(reinterpret_cast<in_t*>(unquant_) + (size_t)row * hidden);
    const auto* xscale = reinterpret_cast<const float*>(xscale_);

    // Pre-norm value s = x (+ residual). Cache in registers; overflow reloads
    // from the pre-norm buffer (res_v when fused-add, else in_v).
    constexpr int CACHE_V = 4;
    Vi cache[CACHE_V];
    float acc = 0.0f;

    auto load_s = [&](int idx) -> Vi {
        Vi x = in_v[idx];
        if(fused_add)
        {
            Vi r = res_v[idx];
#pragma unroll
            for(int j = 0; j < width; ++j)
                x[j] = from_f32<in_t>(to_f32<in_t>(x[j]) + to_f32<in_t>(r[j]));
            res_v[idx] = x; // pre-norm residual write-back
        }
        return x;
    };
    auto sumsq = [&](Vi s) {
#pragma unroll
        for(int j = 0; j < width; ++j)
        {
            float f = to_f32<in_t>(s[j]);
            acc += f * f;
        }
    };
#pragma unroll
    for(int k = 0; k < CACHE_V; ++k)
    {
        const int idx = tid + k * nthreads;
        if(idx < vec_hidden)
        {
            cache[k] = load_s(idx);
            sumsq(cache[k]);
        }
    }
    for(int idx = tid + CACHE_V * nthreads; idx < vec_hidden; idx += nthreads)
        sumsq(load_s(idx));

    float var = block_reduce_sum(acc);
    float inv = rsqrtf(var / hidden + epsilon);

    // normalized value at (row, idx*width+j): n = (t5 ? round(x*inv) : x*inv) * w [* xscale]
    auto norm_j = [&](in_t sval, in_t wval, int col) -> float {
        float xi = to_f32<in_t>(sval) * inv;
        if(t5)
            xi = to_f32<in_t>(from_f32<in_t>(xi));
        float n = xi * to_f32<in_t>(wval);
        if(smooth)
            n *= xscale[col];
        return n;
    };

    // rowmax(|n|)
    float m         = 0.0f;
    auto row_absmax = [&](Vi s, int idx) {
        Vi w = w_v[idx];
#pragma unroll
        for(int j = 0; j < width; ++j)
            m = fmaxf(m, fabsf(norm_j(s[j], w[j], idx * width + j)));
    };
#pragma unroll
    for(int k = 0; k < CACHE_V; ++k)
    {
        const int idx = tid + k * nthreads;
        if(idx < vec_hidden)
            row_absmax(cache[k], idx);
    }
    for(int idx = tid + CACHE_V * nthreads; idx < vec_hidden; idx += nthreads)
        row_absmax(res_v[idx], idx);

    float rowmax = block_reduce_max(m);
    float yscale = rowmax / qmax;
    float inv_ys = yscale > 0.0f ? 1.0f / yscale : 0.0f;
    if(tid == 0)
        reinterpret_cast<float*>(yscale_)[row] = yscale;

    // quantize (and optionally store pre-quant y)
    auto quant = [&](Vi s, int idx) {
        Vi w = w_v[idx];
        Vo q;
        Vi uq;
#pragma unroll
        for(int j = 0; j < width; ++j)
        {
            float n = norm_j(s[j], w[j], idx * width + j);
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
        const int idx = tid + k * nthreads;
        if(idx < vec_hidden)
            quant(cache[k], idx);
    }
    for(int idx = tid + CACHE_V * nthreads; idx < vec_hidden; idx += nthreads)
        quant(res_v[idx], idx);
}

#endif // __HIP_DEVICE_COMPILE__

} // namespace rmsnorm_opus
} // namespace aiter
