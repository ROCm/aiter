// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <c10/cuda/CUDAGuard.h>
#include "dispatch_utils.h"

// =====================================================================================================================
// Kernel Functionalities
//

template <typename scalar_t, typename scalar_f_t>
__device__
void kn_rope_group_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d, const int32_t size_f,
    const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int32_t size_half_f   = size_f >> 1;
    const int32_t offset_half_f = size_half_f * stride_i_d;

    #pragma unroll
    for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
    {
        const int32_t offset_i_d = did * stride_i_d;
        const int32_t offset_o_d = did * stride_o_d;

        float cos, sin;
        sincosf(float(p_freqs[did]), &sin, &cos);

        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_i = hid * stride_i_h + offset_i_d;
            const int32_t offset_o = hid * stride_o_h + offset_o_d;

            const float input = float(p_input[offset_i]);
            const float input_rotate =
                (did < size_half_f) ? float(-p_input[offset_i + offset_half_f]):
                                      float( p_input[offset_i - offset_half_f]);

            p_output[offset_o] = scalar_t(input * cos + input_rotate * sin);
        }
    }

    // the rest are just forwarded
    if (size_d > size_f)
    {
        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_i = hid * stride_i_h;
            const int32_t offset_o = hid * stride_o_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
            {
                p_output[offset_o + did * stride_o_d] = p_input[offset_i + did * stride_i_d];
            }
        }
    }
}

template <typename scalar_t, typename scalar_f_t>
__device__
void kn_rope_group_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d, const int32_t size_f,
    const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_h, const int32_t stride_i_d)
{
    const int32_t size_half_f   = size_f >> 1;
    const int32_t offset_half_f = size_half_f * stride_i_d;

    #pragma unroll
    for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
    {
        const int32_t offset_o_d = did * stride_o_d;
        const int32_t offset_i_d = did * stride_i_d;

        const float cos = cosf(float(p_freqs[did]));
        const float sin =
            (did < size_half_f) ? sinf(float(p_freqs[did + size_half_f])) :
                                 -sinf(float(p_freqs[did - size_half_f]));

        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_o = hid * stride_o_h + offset_o_d;
            const int32_t offset_i = hid * stride_i_h + offset_i_d;

            const float output_grad = float(p_output_grads[offset_o]);
            const float output_grad_rotate =
                (did < size_half_f) ? float(p_output_grads[offset_o + offset_half_f]):
                                      float(p_output_grads[offset_o - offset_half_f]);

            p_input_grads[offset_i] = scalar_t(output_grad * cos + output_grad_rotate * sin);
        }
    }

    // the rest are just forwarded
    if (size_d > size_f)
    {
        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_o = hid * stride_o_h;
            const int32_t offset_i = hid * stride_i_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
            {
                p_input_grads[offset_i + did * stride_i_d] = p_output_grads[offset_o + did * stride_o_d];
            }
        }
    }
}

template <typename scalar_t, typename scalar_f_t>
__device__
void kn_rope_cached_group_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_h, const int32_t size_d, const int32_t size_f,
    const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int32_t size_half_f   = size_f >> 1;
    const int32_t offset_half_f = size_half_f * stride_i_d;

    #pragma unroll
    for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
    {
        const int32_t offset_i_d = did * stride_i_d;
        const int32_t offset_o_d = did * stride_o_d;

        const float cos = p_cos[did];
        const float sin = p_sin[did];

        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_i = hid * stride_i_h + offset_i_d;
            const int32_t offset_o = hid * stride_o_h + offset_o_d;

            const float input = float(p_input[offset_i]);
            const float input_rotate =
                (did < size_half_f) ? float(-p_input[offset_i + offset_half_f]):
                                      float( p_input[offset_i - offset_half_f]);

            p_output[offset_o] = scalar_t(input * cos + input_rotate * sin);
        }
    }

    // the rest are just forwarded
    if (size_d > size_f)
    {
        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_i = hid * stride_i_h;
            const int32_t offset_o = hid * stride_o_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
            {
                p_output[offset_o + did * stride_o_d] = p_input[offset_i + did * stride_i_d];
            }
        }
    }
}

template <typename scalar_t, typename scalar_f_t>
__device__
void kn_rope_cached_group_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_h, const int32_t size_d, const int32_t size_f,
    const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_h, const int32_t stride_i_d)
{
    const int32_t size_half_f   = size_f >> 1;
    const int32_t offset_half_f = size_half_f * stride_i_d;

    #pragma unroll
    for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
    {
        const int32_t offset_o_d = did * stride_o_d;
        const int32_t offset_i_d = did * stride_i_d;

        const float cos = float(p_cos[did]);
        const float sin = (did < size_half_f) ? float(p_sin[did + size_half_f]) : -float(p_sin[did - size_half_f]);

        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_o = hid * stride_o_h + offset_o_d;
            const int32_t offset_i = hid * stride_i_h + offset_i_d;

            const float output_grad = float(p_output_grads[offset_o]);
            const float output_grad_rotate =
                (did < size_half_f) ? float(p_output_grads[offset_o + offset_half_f]):
                                      float(p_output_grads[offset_o - offset_half_f]);

            p_input_grads[offset_i] = scalar_t(output_grad * cos + output_grad_rotate * sin);
        }
    }

    // the rest are just forwarded
    if (size_d > size_f)
    {
        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
        {
            const int32_t offset_o = hid * stride_o_h;
            const int32_t offset_i = hid * stride_i_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
            {
                p_input_grads[offset_i + did * stride_i_d] = p_output_grads[offset_o + did * stride_o_d];
            }
        }
    }
}

// =====================================================================================================================
// Kernel Entries
//

template <typename scalar_t, typename scalar_f_t>
__global__
void kn_rope_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int32_t sid = blockIdx.x;
    const int32_t bid = blockIdx.y;
    const int32_t offset_i = sid * stride_i_s + bid * stride_i_b;
    const int32_t offset_o = sid * stride_o_s + bid * stride_o_b;
    const int32_t offset_f = sid * size_f;

    kn_rope_group_fwd(
        p_output + offset_o,
        p_input + offset_i,
        p_freqs + offset_f,
        size_h, size_d, size_f,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename scalar_t, typename scalar_f_t>
__global__
void kn_rope_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const int32_t sid = blockIdx.x;
    const int32_t bid = blockIdx.y;
    const int32_t offset_o = sid * stride_o_s + bid * stride_o_b;
    const int32_t offset_i = sid * stride_i_s + bid * stride_i_b;
    const int32_t offset_f = sid * size_f;

    kn_rope_group_bwd(
        p_input_grads + offset_i,
        p_output_grads + offset_o,
        p_freqs + offset_f,
        size_h, size_d, size_f,
        stride_o_h, stride_o_d,
        stride_i_h, stride_i_d);
}

template <typename scalar_t, typename scalar_f_t>
__global__
void kn_rope_cached_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int32_t sid = blockIdx.x;
    const int32_t bid = blockIdx.y;
    const int32_t offset_i = sid * stride_i_s + bid * stride_i_b;
    const int32_t offset_o = sid * stride_o_s + bid * stride_o_b;
    const int32_t offset_f = sid * size_f;

    kn_rope_cached_group_fwd(
        p_output + offset_o,
        p_input + offset_i,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h, size_d, size_f,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename scalar_t, typename scalar_f_t>
__global__
void kn_rope_cached_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const int32_t sid = blockIdx.x;
    const int32_t bid = blockIdx.y;
    const int32_t offset_o = sid * stride_o_s + bid * stride_o_b;
    const int32_t offset_i = sid * stride_i_s + bid * stride_i_b;
    const int32_t offset_f = sid * size_f;

    kn_rope_cached_group_bwd(
        p_input_grads + offset_i,
        p_output_grads + offset_o,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h, size_d, size_f,
        stride_o_h, stride_o_d,
        stride_i_h, stride_i_d);
}

template <typename scalar_t, typename scalar_f_t>
__global__
void kn_rope_thd_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const int32_t* __restrict__    p_cu_seqlens,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_i_t, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_t, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int32_t sid = blockIdx.x;
    const int32_t bid = blockIdx.y;
    const int32_t tid = sid + p_cu_seqlens[bid];

    if (tid < p_cu_seqlens[bid + 1])
    {
        const int32_t offset_i = tid * stride_i_t;
        const int32_t offset_o = tid * stride_o_t;
        const int32_t offset_f = sid * size_f;

        kn_rope_group_fwd(
            p_output + offset_o,
            p_input + offset_i,
            p_freqs + offset_f,
            size_h, size_d, size_f,
            stride_i_h, stride_i_d,
            stride_o_h, stride_o_d);
    }
}

template <typename scalar_t, typename scalar_f_t>
__global__
void kn_rope_thd_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const int32_t* __restrict__    p_cu_seqlens,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_o_t, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_t, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const int32_t sid = blockIdx.x;
    const int32_t bid = blockIdx.y;
    const int32_t tid = sid + p_cu_seqlens[bid];

    if (tid < p_cu_seqlens[bid + 1])
    {
        const int32_t offset_o = tid * stride_o_t;
        const int32_t offset_i = tid * stride_i_t;
        const int32_t offset_f = sid * size_f;

        kn_rope_group_bwd(
            p_input_grads + offset_i,
            p_output_grads + offset_o,
            p_freqs + offset_f,
            size_h, size_d, size_f,
            stride_o_h, stride_o_d,
            stride_i_h, stride_i_d);
    }
}

template <typename scalar_t, typename scalar_f_t>
__global__
void kn_rope_2d_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_cos_h,
    const scalar_f_t* __restrict__ p_sin_h,
    const scalar_f_t* __restrict__ p_cos_w,
    const scalar_f_t* __restrict__ p_sin_w,
    const int32_t size_W, const int32_t size_h, const int32_t size_d,
    const int32_t stride_i_b, const int32_t stride_i_H, const int32_t stride_i_W, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_b, const int32_t stride_o_s, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int Hid = blockIdx.x;
    const int Wid = blockIdx.y;
    const int sid = Hid * size_W + Wid;
    const int bid = blockIdx.z;
    const int size_half_d = size_d >> 1;

    const int offset_h_i = bid * stride_i_b + Hid * stride_i_H + Wid * stride_i_W;
    const int offset_h_o = bid * stride_o_b + sid * stride_o_s;
    const int offset_h_f = Hid * size_half_d;
    kn_rope_cached_group_fwd(
        p_output + offset_h_o,
        p_input + offset_h_i,
        p_cos_h + offset_h_f,
        p_sin_h + offset_h_f,
        size_h, size_half_d, size_half_d,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);

    const int offset_w_i = offset_h_i + size_half_d * stride_i_d;
    const int offset_w_o = offset_h_o + size_half_d * stride_o_d;
    const int offset_w_f = Wid * size_half_d;
    kn_rope_cached_group_fwd(
        p_output + offset_w_o,
        p_input + offset_w_i,
        p_cos_w + offset_w_f,
        p_sin_w + offset_w_f,
        size_h, size_half_d, size_half_d,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename scalar_t, typename scalar_f_t>
__global__
void kn_rope_2d_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const scalar_f_t* __restrict__ p_cos_h,
    const scalar_f_t* __restrict__ p_sin_h,
    const scalar_f_t* __restrict__ p_cos_w,
    const scalar_f_t* __restrict__ p_sin_w,
    const int32_t size_W, const int32_t size_h, const int32_t size_d,
    const int32_t stride_o_b, const int32_t stride_o_H, const int32_t stride_o_W, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_b, const int32_t stride_i_s, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const int Hid = blockIdx.x;
    const int Wid = blockIdx.y;
    const int sid = Hid * size_W + Wid;
    const int bid = blockIdx.z;
    const int size_half_d = size_d >> 1;

    const int offset_h_o = bid * stride_o_b + Hid * stride_o_H + Wid * stride_o_W;
    const int offset_h_i = bid * stride_i_b + sid * stride_i_s;
    const int offset_h_f = Hid * size_half_d;
    kn_rope_cached_group_bwd(
        p_input_grads + offset_h_i,
        p_output_grads + offset_h_o,
        p_cos_h + offset_h_f,
        p_sin_h + offset_h_f,
        size_h, size_half_d, size_half_d,
        stride_o_h, stride_o_d,
        stride_i_h, stride_i_d);

    const int offset_w_o = offset_h_o + size_half_d * stride_o_d;
    const int offset_w_i = offset_h_i + size_half_d * stride_i_d;
    const int offset_w_f = Wid * size_half_d;
    kn_rope_cached_group_bwd(
        p_input_grads + offset_w_i,
        p_output_grads + offset_w_o,
        p_cos_w + offset_w_f,
        p_sin_w + offset_w_f,
        size_h, size_half_d, size_half_d,
        stride_o_h, stride_o_d,
        stride_i_h, stride_i_d);
}

// =====================================================================================================================
// Dispatches
//

template <typename scalar_t, typename scalar_f_t>
void dispatch_rope_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_fwd<<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_freqs,
        size_h, size_d, size_f,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d);
}

template <typename scalar_t, typename scalar_f_t>
void dispatch_rope_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_bwd<<<grid, block, 0, stream>>>(
        p_input_grads,
        p_output_grads,
        p_freqs,
        size_h, size_d, size_f,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d);
}

template <typename scalar_t, typename scalar_f_t>
void dispatch_rope_cached_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_cached_fwd<<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_cos, p_sin,
        size_h, size_d, size_f,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d);
}

template <typename scalar_t, typename scalar_f_t>
void dispatch_rope_cached_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_cached_bwd<<<grid, block, 0, stream>>>(
        p_input_grads,
        p_output_grads,
        p_cos, p_sin,
        size_h, size_d, size_f,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d);
}

template <typename scalar_t, typename scalar_f_t>
void dispatch_rope_thd_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const int32_t* __restrict__    p_cu_seqlens,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_max_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_i_t, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_t, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_max_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_thd_fwd<<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_cu_seqlens,
        p_freqs,
        size_h, size_d, size_f,
        stride_i_t, stride_i_h, stride_i_d,
        stride_o_t, stride_o_h, stride_o_d);
}

template <typename scalar_t, typename scalar_f_t>
void dispatch_rope_thd_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const int32_t* __restrict__    p_cu_seqlens,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_max_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_o_t, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_t, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_max_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_thd_bwd<<<grid, block, 0, stream>>>(
        p_input_grads,
        p_output_grads,
        p_cu_seqlens,
        p_freqs,
        size_h, size_d, size_f,
        stride_o_t, stride_o_h, stride_o_d,
        stride_i_t, stride_i_h, stride_i_d);
}

template <typename scalar_t, typename scalar_f_t>
void dispatch_rope_2d_fwd(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_cos_h,
    const scalar_f_t* __restrict__ p_sin_h,
    const scalar_f_t* __restrict__ p_cos_w,
    const scalar_f_t* __restrict__ p_sin_w,
    const int32_t size_b, const int32_t size_H, const int32_t size_W, const int32_t size_h, const int32_t size_d,
    const int32_t stride_i_b, const int32_t stride_i_H, const int32_t stride_i_W, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_b, const int32_t stride_o_s, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_H, size_W, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_2d_fwd<<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_cos_h, p_sin_h,
        p_cos_w, p_sin_w,
        size_W, size_h, size_d,
        stride_i_b, stride_i_H, stride_i_W, stride_i_h, stride_i_d,
        stride_o_b, stride_o_s, stride_o_h, stride_o_d);
}

template <typename scalar_t, typename scalar_f_t>
void dispatch_rope_2d_bwd(
    scalar_t* __restrict__         p_input_grads,
    const scalar_t* __restrict__   p_output_grads,
    const scalar_f_t* __restrict__ p_cos_h,
    const scalar_f_t* __restrict__ p_sin_h,
    const scalar_f_t* __restrict__ p_cos_w,
    const scalar_f_t* __restrict__ p_sin_w,
    const int32_t size_b, const int32_t size_H, const int32_t size_W, const int32_t size_h, const int32_t size_d,
    const int32_t stride_o_b, const int32_t stride_o_H, const int32_t stride_o_W, const int32_t stride_o_h, const int32_t stride_o_d,
    const int32_t stride_i_b, const int32_t stride_i_s, const int32_t stride_i_h, const int32_t stride_i_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_H, size_W, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_rope_2d_bwd<<<grid, block, 0, stream>>>(
        p_input_grads,
        p_output_grads,
        p_cos_h, p_sin_h,
        p_cos_w, p_sin_w,
        size_W, size_h, size_d,
        stride_o_b, stride_o_H, stride_o_W, stride_o_h, stride_o_d,
        stride_i_b, stride_i_s, stride_i_h, stride_i_d);
}

// =====================================================================================================================
// Interfaces
//

void rope_fwd_impl(
    torch::Tensor&       output,        // [s, b, h, d]
    const torch::Tensor& input,         // [s, b, h, d]
    const torch::Tensor& freqs)         // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = input.size(0);
    const int32_t size_b = input.size(1);
    const int32_t size_h = input.size(2);
    const int32_t size_d = input.size(3);
    const int32_t size_f = freqs.size(3);
    // Get strides of input
    const int32_t stride_i_s = input.stride(0);
    const int32_t stride_i_b = input.stride(1);
    const int32_t stride_i_h = input.stride(2);
    const int32_t stride_i_d = input.stride(3);
    // Get strides of output
    const int32_t stride_o_s = output.stride(0);
    const int32_t stride_o_b = output.stride(1);
    const int32_t stride_o_h = output.stride(2);
    const int32_t stride_o_d = output.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "kn_rope_fwd",
        [&] {
            switch (freqs.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_rope_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    freqs.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::Half:
                dispatch_rope_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_rope_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::BFloat16>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_fwd doesn't support to specified formats.");
                break;
            }
        });
}

void rope_bwd_impl(
    torch::Tensor&       input_grads,   // [s, b, h, d]
    const torch::Tensor& output_grads,  // [s, b, h, d]
    const torch::Tensor& freqs)         // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = output_grads.size(0);
    const int32_t size_b = output_grads.size(1);
    const int32_t size_h = output_grads.size(2);
    const int32_t size_d = output_grads.size(3);
    const int32_t size_f = freqs.size(3);
    // Get strides of output_grads
    const int32_t stride_o_s = output_grads.stride(0);
    const int32_t stride_o_b = output_grads.stride(1);
    const int32_t stride_o_h = output_grads.stride(2);
    const int32_t stride_o_d = output_grads.stride(3);
    // Get strides of input_grads
    const int32_t stride_i_s = input_grads.stride(0);
    const int32_t stride_i_b = input_grads.stride(1);
    const int32_t stride_i_h = input_grads.stride(2);
    const int32_t stride_i_d = input_grads.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        output_grads.scalar_type(),
        "kn_rope_bwd",
        [&] {
            switch (freqs.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_rope_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    freqs.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d,
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::Half:
                dispatch_rope_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d,
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_rope_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::BFloat16>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d,
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_bwd doesn't support to specified formats.");
                break;
            }
        });
}

void rope_cached_fwd_impl(
    torch::Tensor&       output,        // [s, b, h, d]
    const torch::Tensor& input,         // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin)           // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = input.size(0);
    const int32_t size_b = input.size(1);
    const int32_t size_h = input.size(2);
    const int32_t size_d = input.size(3);
    const int32_t size_f = cos.size(3);
    // Get strides of input
    const int32_t stride_i_s = input.stride(0);
    const int32_t stride_i_b = input.stride(1);
    const int32_t stride_i_h = input.stride(2);
    const int32_t stride_i_d = input.stride(3);
    // Get strides of output
    const int32_t stride_o_s = output.stride(0);
    const int32_t stride_o_b = output.stride(1);
    const int32_t stride_o_h = output.stride(2);
    const int32_t stride_o_d = output.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "kn_rope_cached_fwd",
        [&] {
            switch (cos.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_rope_cached_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cos.data_ptr<float>(),
                    sin.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::Half:
                dispatch_rope_cached_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cos.data_ptr<at::Half>(),
                    sin.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_rope_cached_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cos.data_ptr<at::BFloat16>(),
                    sin.data_ptr<at::BFloat16>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_cached_fwd doesn't support to specified formats.");
                break;
            }
        });
}

void rope_cached_bwd_impl(
    torch::Tensor&       input_grads,   // [s, b, h, d]
    const torch::Tensor& output_grads,  // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin)           // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = output_grads.size(0);
    const int32_t size_b = output_grads.size(1);
    const int32_t size_h = output_grads.size(2);
    const int32_t size_d = output_grads.size(3);
    const int32_t size_f = cos.size(3);
    // Get strides of output_grads
    const int32_t stride_o_s = output_grads.stride(0);
    const int32_t stride_o_b = output_grads.stride(1);
    const int32_t stride_o_h = output_grads.stride(2);
    const int32_t stride_o_d = output_grads.stride(3);
    // Get strides of input_grads
    const int32_t stride_i_s = input_grads.stride(0);
    const int32_t stride_i_b = input_grads.stride(1);
    const int32_t stride_i_h = input_grads.stride(2);
    const int32_t stride_i_d = input_grads.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        output_grads.scalar_type(),
        "kn_rope_cached_bwd",
        [&] {
            switch (cos.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_rope_cached_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cos.data_ptr<float>(),
                    sin.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d,
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::Half:
                dispatch_rope_cached_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cos.data_ptr<at::Half>(),
                    sin.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d,
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_rope_cached_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cos.data_ptr<at::BFloat16>(),
                    sin.data_ptr<at::BFloat16>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d,
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_cached_bwd doesn't support to specified formats.");
                break;
            }
        });
}

void rope_thd_fwd_impl(
    torch::Tensor&       output,        // [t, h, d]
    const torch::Tensor& input,         // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [b + 1]
    const torch::Tensor& freqs)         // [max_s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_h     = input.size(1);
    const int32_t size_d     = input.size(2);
    const int32_t size_f     = freqs.size(3);
    const int32_t size_b     = cu_seqlens.size(0) - 1;
    const int32_t size_max_s = freqs.size(0);
    // Get strides of input
    const int32_t stride_i_t = input.stride(0);
    const int32_t stride_i_h = input.stride(1);
    const int32_t stride_i_d = input.stride(2);
    // Get strides of output
    const int32_t stride_o_t = output.stride(0);
    const int32_t stride_o_h = output.stride(1);
    const int32_t stride_o_d = output.stride(2);

    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "kn_rope_thd_fwd",
        [&] {
            switch (freqs.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_rope_thd_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cu_seqlens.data_ptr<int32_t>(),
                    freqs.data_ptr<float>(),
                    size_max_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_t, stride_i_h, stride_i_d,
                    stride_o_t, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::Half:
                dispatch_rope_thd_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cu_seqlens.data_ptr<int32_t>(),
                    freqs.data_ptr<at::Half>(),
                    size_max_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_t, stride_i_h, stride_i_d,
                    stride_o_t, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_rope_thd_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cu_seqlens.data_ptr<int32_t>(),
                    freqs.data_ptr<at::BFloat16>(),
                    size_max_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_t, stride_i_h, stride_i_d,
                    stride_o_t, stride_o_h, stride_o_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_thd_fwd doesn't support to specified formats.");
                break;
            }
        });
}

void rope_thd_bwd_impl(
    torch::Tensor&       input_grads,   // [t, h, d]
    const torch::Tensor& output_grads,  // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [b + 1]
    const torch::Tensor& freqs)         // [max_s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_h     = output_grads.size(1);
    const int32_t size_d     = output_grads.size(2);
    const int32_t size_f     = freqs.size(3);
    const int32_t size_b     = cu_seqlens.size(0) - 1;
    const int32_t size_max_s = freqs.size(0);
    // Get strides of output_grads
    const int32_t stride_o_t = output_grads.stride(0);
    const int32_t stride_o_h = output_grads.stride(1);
    const int32_t stride_o_d = output_grads.stride(2);
    // Get strides of input_grads
    const int32_t stride_i_t = input_grads.stride(0);
    const int32_t stride_i_h = input_grads.stride(1);
    const int32_t stride_i_d = input_grads.stride(2);

    VLLM_DISPATCH_FLOATING_TYPES(
        output_grads.scalar_type(),
        "kn_rope_thd_bwd",
        [&] {
            switch (freqs.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_rope_thd_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cu_seqlens.data_ptr<int32_t>(),
                    freqs.data_ptr<float>(),
                    size_max_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_t, stride_o_h, stride_o_d,
                    stride_i_t, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::Half:
                dispatch_rope_thd_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cu_seqlens.data_ptr<int32_t>(),
                    freqs.data_ptr<at::Half>(),
                    size_max_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_t, stride_o_h, stride_o_d,
                    stride_i_t, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_rope_thd_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cu_seqlens.data_ptr<int32_t>(),
                    freqs.data_ptr<at::BFloat16>(),
                    size_max_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_t, stride_o_h, stride_o_d,
                    stride_i_t, stride_i_h, stride_i_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_thd_bwd doesn't support to specified formats.");
                break;
            }
        });
}

void rope_2d_fwd_impl(
    torch::Tensor&       output,
    const torch::Tensor& input,
    const torch::Tensor& cos_h,
    const torch::Tensor& sin_h,
    const torch::Tensor& cos_w,
    const torch::Tensor& sin_w)
{
    // Get sizes of input and output
    const int size_b = input.size(0);
    const int size_H = input.size(1);
    const int size_W = input.size(2);
    const int size_h = input.size(3);
    const int size_d = input.size(4);
    // Get strides of input
    const int stride_i_b = input.stride(0);
    const int stride_i_H = input.stride(1);
    const int stride_i_W = input.stride(2);
    const int stride_i_h = input.stride(3);
    const int stride_i_d = input.stride(4);
    // Get strides of output
    const int stride_o_b = output.stride(0);
    const int stride_o_s = output.stride(1);
    const int stride_o_h = output.stride(2);
    const int stride_o_d = output.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "kn_rope_2d_fwd",
        [&] {
            switch (cos_h.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_rope_2d_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cos_h.data_ptr<float>(),
                    sin_h.data_ptr<float>(),
                    cos_w.data_ptr<float>(),
                    sin_w.data_ptr<float>(),
                    size_b, size_H, size_W, size_h, size_d,
                    stride_i_b, stride_i_H, stride_i_W, stride_i_h, stride_i_d,
                    stride_o_b, stride_o_s, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::Half:
                dispatch_rope_2d_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cos_h.data_ptr<at::Half>(),
                    sin_h.data_ptr<at::Half>(),
                    cos_w.data_ptr<at::Half>(),
                    sin_w.data_ptr<at::Half>(),
                    size_b, size_H, size_W, size_h, size_d,
                    stride_i_b, stride_i_H, stride_i_W, stride_i_h, stride_i_d,
                    stride_o_b, stride_o_s, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_rope_2d_fwd(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cos_h.data_ptr<at::BFloat16>(),
                    sin_h.data_ptr<at::BFloat16>(),
                    cos_w.data_ptr<at::BFloat16>(),
                    sin_w.data_ptr<at::BFloat16>(),
                    size_b, size_H, size_W, size_h, size_d,
                    stride_i_b, stride_i_H, stride_i_W, stride_i_h, stride_i_d,
                    stride_o_b, stride_o_s, stride_o_h, stride_o_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_2d_fwd doesn't support to specified formats.");
                break;
            }
        });
}

void rope_2d_bwd_impl(
    torch::Tensor&       input_grads,
    const torch::Tensor& output_grads,
    const torch::Tensor& cos_h,
    const torch::Tensor& sin_h,
    const torch::Tensor& cos_w,
    const torch::Tensor& sin_w)
{
    // Get sizes of input and output
    const int size_b = output_grads.size(0);
    const int size_H = output_grads.size(1);
    const int size_W = output_grads.size(2);
    const int size_h = output_grads.size(3);
    const int size_d = output_grads.size(4);
    // Get strides of output_grads
    const int stride_o_b = output_grads.stride(0);
    const int stride_o_H = output_grads.stride(1);
    const int stride_o_W = output_grads.stride(2);
    const int stride_o_h = output_grads.stride(3);
    const int stride_o_d = output_grads.stride(4);
    // Get strides of input_grads
    const int stride_i_b = input_grads.stride(0);
    const int stride_i_s = input_grads.stride(1);
    const int stride_i_h = input_grads.stride(2);
    const int stride_i_d = input_grads.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        output_grads.scalar_type(),
        "kn_rope_2d_bwd",
        [&] {
            switch (cos_h.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_rope_2d_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cos_h.data_ptr<float>(),
                    sin_h.data_ptr<float>(),
                    cos_w.data_ptr<float>(),
                    sin_w.data_ptr<float>(),
                    size_b, size_H, size_W, size_h, size_d,
                    stride_o_b, stride_o_H, stride_o_W, stride_o_h, stride_o_d,
                    stride_i_b, stride_i_s, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::Half:
                dispatch_rope_2d_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cos_h.data_ptr<at::Half>(),
                    sin_h.data_ptr<at::Half>(),
                    cos_w.data_ptr<at::Half>(),
                    sin_w.data_ptr<at::Half>(),
                    size_b, size_H, size_W, size_h, size_d,
                    stride_o_b, stride_o_H, stride_o_W, stride_o_h, stride_o_d,
                    stride_i_b, stride_i_s, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_rope_2d_bwd(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cos_h.data_ptr<at::BFloat16>(),
                    sin_h.data_ptr<at::BFloat16>(),
                    cos_w.data_ptr<at::BFloat16>(),
                    sin_w.data_ptr<at::BFloat16>(),
                    size_b, size_H, size_W, size_h, size_d,
                    stride_o_b, stride_o_H, stride_o_W, stride_o_h, stride_o_d,
                    stride_i_b, stride_i_s, stride_i_h, stride_i_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_2d_bwd doesn't support to specified formats.");
                break;
            }
        });
}
