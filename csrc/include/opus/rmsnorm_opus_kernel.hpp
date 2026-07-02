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

// out[t,i] = scalar( f32(in[t,i]) * rsqrt(mean_i(in[t,:]^2) + eps) * f32(w[i]) )
template <typename scalar_t, int width>
__global__ void rmsnorm2d_fwd_kernel(void* __restrict__ out,
                                     const void* __restrict__ in,
                                     const void* __restrict__ weight,
                                     float epsilon,
                                     int rows,
                                     int hidden);

// Fused residual add + rmsnorm, in place:
//   x = in + residual;  residual = x;  in = rmsnorm(x) * weight
template <typename scalar_t, int width>
__global__ void fused_add_rmsnorm2d_fwd_kernel(void* __restrict__ inout,
                                               void* __restrict__ residual,
                                               const void* __restrict__ weight,
                                               float epsilon,
                                               int rows,
                                               int hidden);

#if !defined(__HIP_DEVICE_COMPILE__)
// Host pass: empty stubs so the __device_stub__ symbols resolve.
template <typename scalar_t, int width>
__global__ void rmsnorm2d_fwd_kernel(void*, const void*, const void*, float, int, int)
{
}
template <typename scalar_t, int width>
__global__ void fused_add_rmsnorm2d_fwd_kernel(void*, void*, const void*, float, int, int)
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

// Butterfly warp reduction over a wavefront (wave32 or wave64).
__device__ inline float warp_reduce_sum(float v)
{
    constexpr int ws = opus::get_warp_size();
#pragma unroll
    for(int o = ws / 2; o > 0; o >>= 1)
        v += opus::shfl(v, opus::lane_id() ^ o);
    return v;
}

// Block reduction correct for any blockDim (not only powers of two).
__device__ inline float block_reduce_sum(float v)
{
    constexpr int ws        = opus::get_warp_size();
    constexpr int max_warps = 1024 / ws;
    __shared__ float s_warp[max_warps];

    const int tid  = opus::thread_id_x();
    const int lane = tid % ws;
    const int wid  = tid / ws;

    v = warp_reduce_sum(v);
    if(lane == 0)
        s_warp[wid] = v;
    opus::sync_threads();

    const int num_warps = (opus::block_size_x() + ws - 1) / ws;
    v                   = (tid < num_warps) ? s_warp[lane] : 0.0f;
    if(wid == 0)
        v = warp_reduce_sum(v);

    __shared__ float s_bcast;
    if(tid == 0)
        s_bcast = v;
    opus::sync_threads();
    return s_bcast;
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
                                     int hidden)
{
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

    float acc = 0.0f;
    for(int idx = tid; idx < vec_hidden; idx += nthreads)
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

    for(int idx = tid; idx < vec_hidden; idx += nthreads)
    {
        V x = in_v[idx];
        V w = w_v[idx];
        V y;
#pragma unroll
        for(int j = 0; j < width; ++j)
            y[j] = from_f32<scalar_t>(to_f32<scalar_t>(x[j]) * inv * to_f32<scalar_t>(w[j]));
        out_v[idx] = y;
    }
}

template <typename scalar_t, int width>
__global__ void fused_add_rmsnorm2d_fwd_kernel(void* __restrict__ inout_,
                                               void* __restrict__ residual_,
                                               const void* __restrict__ weight_,
                                               float epsilon,
                                               int rows,
                                               int hidden)
{
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

    float acc = 0.0f;
    for(int idx = tid; idx < vec_hidden; idx += nthreads)
    {
        V x = io_v[idx];
        V r = res_v[idx];
        V s;
#pragma unroll
        for(int j = 0; j < width; ++j)
        {
            float f = to_f32<scalar_t>(x[j]) + to_f32<scalar_t>(r[j]);
            s[j]    = from_f32<scalar_t>(f);
            acc += f * f;
        }
        res_v[idx] = s; // pre-norm residual write-back
    }

    float var = block_reduce_sum(acc);
    float inv = rsqrtf(var / hidden + epsilon);

    for(int idx = tid; idx < vec_hidden; idx += nthreads)
    {
        V s = res_v[idx];
        V w = w_v[idx];
        V y;
#pragma unroll
        for(int j = 0; j < width; ++j)
            y[j] = from_f32<scalar_t>(to_f32<scalar_t>(s[j]) * inv * to_f32<scalar_t>(w[j]));
        io_v[idx] = y;
    }
}

#endif // __HIP_DEVICE_COMPILE__

} // namespace rmsnorm_opus
} // namespace aiter
