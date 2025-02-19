// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <c10/cuda/CUDAGuard.h>
#include "dispatch_utils.h"

// =====================================================================================================================
// Kernel Functionalities
//

struct Op1cUncachedFwd
{
    template <typename scalar_t, typename scalar_f_t>
    __device__ void operator()(
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
};

struct Op1cUncachedBwd
{
    template <typename scalar_t, typename scalar_f_t>
    __device__ void operator()(
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
};

struct Op2cUncachedFwd
{
    template <typename scalar_t, typename scalar_f_t>
    __device__ void operator()(
        scalar_t* __restrict__         p_output_x,
        scalar_t* __restrict__         p_output_y,
        const scalar_t* __restrict__   p_input_x,
        const scalar_t* __restrict__   p_input_y,
        const scalar_f_t* __restrict__ p_freqs,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_ix_h, const int32_t stride_ix_d,
        const int32_t stride_iy_h, const int32_t stride_iy_d,
        const int32_t stride_ox_h, const int32_t stride_ox_d,
        const int32_t stride_oy_h, const int32_t stride_oy_d)
    {
        const int32_t size_half_f   = size_f >> 1;
        const int32_t offset_half_f_x = size_half_f * stride_ix_d;
        const int32_t offset_half_f_y = size_half_f * stride_iy_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
        {
            const int32_t offset_ix_d = did * stride_ix_d;
            const int32_t offset_iy_d = did * stride_iy_d;
            const int32_t offset_ox_d = did * stride_ox_d;
            const int32_t offset_oy_d = did * stride_oy_d;

            float cos, sin;
            sincosf(float(p_freqs[did]), &sin, &cos);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ix = hid * stride_ix_h + offset_ix_d;
                const int32_t offset_iy = hid * stride_iy_h + offset_iy_d;
                const int32_t offset_ox = hid * stride_ox_h + offset_ox_d;
                const int32_t offset_oy = hid * stride_oy_h + offset_oy_d;

                const float input_x = float(p_input_x[offset_ix]);
                const float input_y = float(p_input_y[offset_iy]);
                const float input_x_rotate =
                    (did < size_half_f) ? float(-p_input_x[offset_ix + offset_half_f_x]):
                                        float( p_input_x[offset_ix - offset_half_f_x]);
                const float input_y_rotate =
                    (did < size_half_f) ? float(-p_input_y[offset_iy + offset_half_f_y]):
                                        float( p_input_y[offset_iy - offset_half_f_y]);

                p_output_x[offset_ox] = scalar_t(input_x * cos + input_x_rotate * sin);
                p_output_y[offset_oy] = scalar_t(input_y * cos + input_y_rotate * sin);
            }
        }

        // the rest are just forwarded
        if (size_d > size_f)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ix = hid * stride_ix_h;
                const int32_t offset_iy = hid * stride_iy_h;
                const int32_t offset_ox = hid * stride_ox_h;
                const int32_t offset_oy = hid * stride_oy_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
                {
                    p_output_x[offset_ox + did * stride_ox_d] = p_input_x[offset_ix + did * stride_ix_d];
                    p_output_y[offset_oy + did * stride_oy_d] = p_input_y[offset_iy + did * stride_iy_d];
                }
            }
        }
    }
};

struct Op2cUncachedBwd
{
    template <typename scalar_t, typename scalar_f_t>
    __device__ void operator()(
        scalar_t* __restrict__         p_input_grads_x,
        scalar_t* __restrict__         p_input_grads_y,
        const scalar_t* __restrict__   p_output_grads_x,
        const scalar_t* __restrict__   p_output_grads_y,
        const scalar_f_t* __restrict__ p_freqs,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_ox_h, const int32_t stride_ox_d,
        const int32_t stride_oy_h, const int32_t stride_oy_d,
        const int32_t stride_ix_h, const int32_t stride_ix_d,
        const int32_t stride_iy_h, const int32_t stride_iy_d)
    {
        const int32_t size_half_f   = size_f >> 1;
        const int32_t offset_half_f_x = size_half_f * stride_ix_d;
        const int32_t offset_half_f_y = size_half_f * stride_iy_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
        {
            const int32_t offset_ox_d = did * stride_ox_d;
            const int32_t offset_oy_d = did * stride_oy_d;
            const int32_t offset_ix_d = did * stride_ix_d;
            const int32_t offset_iy_d = did * stride_iy_d;

            const float cos = cosf(float(p_freqs[did]));
            const float sin =
                (did < size_half_f) ? sinf(float(p_freqs[did + size_half_f])) :
                                    -sinf(float(p_freqs[did - size_half_f]));

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ox = hid * stride_ox_h + offset_ox_d;
                const int32_t offset_oy = hid * stride_oy_h + offset_oy_d;
                const int32_t offset_ix = hid * stride_ix_h + offset_ix_d;
                const int32_t offset_iy = hid * stride_iy_h + offset_iy_d;

                const float output_grad_x = float(p_output_grads_x[offset_ox]);
                const float output_grad_y = float(p_output_grads_y[offset_oy]);
                const float output_grad_x_rotate =
                    (did < size_half_f) ? float(p_output_grads_x[offset_ox + offset_half_f_x]):
                                        float(p_output_grads_x[offset_ox - offset_half_f_x]);
                const float output_grad_y_rotate =
                    (did < size_half_f) ? float(p_output_grads_y[offset_oy + offset_half_f_y]):
                                        float(p_output_grads_y[offset_oy - offset_half_f_y]);

                p_input_grads_x[offset_ix] = scalar_t(output_grad_x * cos + output_grad_x_rotate * sin);
                p_input_grads_y[offset_iy] = scalar_t(output_grad_y * cos + output_grad_y_rotate * sin);
            }
        }

        // the rest are just forwarded
        if (size_d > size_f)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ox = hid * stride_ox_h;
                const int32_t offset_oy = hid * stride_oy_h;
                const int32_t offset_ix = hid * stride_ix_h;
                const int32_t offset_iy = hid * stride_iy_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
                {
                    p_input_grads_x[offset_ix + did * stride_ix_d] = p_output_grads_x[offset_ox + did * stride_ox_d];
                    p_input_grads_y[offset_iy + did * stride_iy_d] = p_output_grads_y[offset_oy + did * stride_oy_d];
                }
            }
        }
    }
};

struct Op1cCachedFwd
{
    template <typename scalar_t, typename scalar_f_t>
    __device__ void operator()(
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
};

struct Op1cCachedBwd
{
    template <typename scalar_t, typename scalar_f_t>
    __device__ void operator()(
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
};

struct Op2cCachedFwd
{
    template <typename scalar_t, typename scalar_f_t>
    __device__ void operator()(
        scalar_t* __restrict__         p_output_x,
        scalar_t* __restrict__         p_output_y,
        const scalar_t* __restrict__   p_input_x,
        const scalar_t* __restrict__   p_input_y,
        const scalar_f_t* __restrict__ p_cos,
        const scalar_f_t* __restrict__ p_sin,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_ix_h, const int32_t stride_ix_d,
        const int32_t stride_iy_h, const int32_t stride_iy_d,
        const int32_t stride_ox_h, const int32_t stride_ox_d,
        const int32_t stride_oy_h, const int32_t stride_oy_d)
    {
        const int32_t size_half_f   = size_f >> 1;
        const int32_t offset_half_f_x = size_half_f * stride_ix_d;
        const int32_t offset_half_f_y = size_half_f * stride_iy_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
        {
            const int32_t offset_ix_d = did * stride_ix_d;
            const int32_t offset_iy_d = did * stride_iy_d;
            const int32_t offset_ox_d = did * stride_ox_d;
            const int32_t offset_oy_d = did * stride_oy_d;

            const float cos = p_cos[did];
            const float sin = p_sin[did];

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ix = hid * stride_ix_h + offset_ix_d;
                const int32_t offset_iy = hid * stride_iy_h + offset_iy_d;
                const int32_t offset_ox = hid * stride_ox_h + offset_ox_d;
                const int32_t offset_oy = hid * stride_oy_h + offset_oy_d;

                const float input_x = float(p_input_x[offset_ix]);
                const float input_y = float(p_input_y[offset_iy]);
                const float input_x_rotate =
                    (did < size_half_f) ? float(-p_input_x[offset_ix + offset_half_f_x]):
                                        float( p_input_x[offset_ix - offset_half_f_x]);
                const float input_y_rotate =
                    (did < size_half_f) ? float(-p_input_y[offset_iy + offset_half_f_y]):
                                        float( p_input_y[offset_iy - offset_half_f_y]);

                p_output_x[offset_ox] = scalar_t(input_x * cos + input_x_rotate * sin);
                p_output_y[offset_oy] = scalar_t(input_y * cos + input_y_rotate * sin);
            }
        }

        // the rest are just forwarded
        if (size_d > size_f)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ix = hid * stride_ix_h;
                const int32_t offset_iy = hid * stride_iy_h;
                const int32_t offset_ox = hid * stride_ox_h;
                const int32_t offset_oy = hid * stride_oy_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
                {
                    p_output_x[offset_ox + did * stride_ox_d] = p_input_x[offset_ix + did * stride_ix_d];
                    p_output_y[offset_oy + did * stride_oy_d] = p_input_y[offset_iy + did * stride_iy_d];
                }
            }
        }
    }
};

struct Op2cCachedBwd
{
    template <typename scalar_t, typename scalar_f_t>
    __device__ void operator()(
        scalar_t* __restrict__         p_input_grads_x,
        scalar_t* __restrict__         p_input_grads_y,
        const scalar_t* __restrict__   p_output_grads_x,
        const scalar_t* __restrict__   p_output_grads_y,
        const scalar_f_t* __restrict__ p_cos,
        const scalar_f_t* __restrict__ p_sin,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_ox_h, const int32_t stride_ox_d,
        const int32_t stride_oy_h, const int32_t stride_oy_d,
        const int32_t stride_ix_h, const int32_t stride_ix_d,
        const int32_t stride_iy_h, const int32_t stride_iy_d)
    {
        const int32_t size_half_f   = size_f >> 1;
        const int32_t offset_half_f_x = size_half_f * stride_ix_d;
        const int32_t offset_half_f_y = size_half_f * stride_iy_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_f; did += blockDim.x)
        {
            const int32_t offset_ox_d = did * stride_ox_d;
            const int32_t offset_oy_d = did * stride_oy_d;
            const int32_t offset_ix_d = did * stride_ix_d;
            const int32_t offset_iy_d = did * stride_iy_d;

            const float cos = float(p_cos[did]);
            const float sin = (did < size_half_f) ? float(p_sin[did + size_half_f]) : -float(p_sin[did - size_half_f]);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ox = hid * stride_ox_h + offset_ox_d;
                const int32_t offset_oy = hid * stride_oy_h + offset_oy_d;
                const int32_t offset_ix = hid * stride_ix_h + offset_ix_d;
                const int32_t offset_iy = hid * stride_iy_h + offset_iy_d;

                const float output_grad_x = float(p_output_grads_x[offset_ox]);
                const float output_grad_y = float(p_output_grads_y[offset_oy]);
                const float output_grad_x_rotate =
                    (did < size_half_f) ? float(p_output_grads_x[offset_ox + offset_half_f_x]):
                                        float(p_output_grads_x[offset_ox - offset_half_f_x]);
                const float output_grad_y_rotate =
                    (did < size_half_f) ? float(p_output_grads_y[offset_oy + offset_half_f_y]):
                                        float(p_output_grads_y[offset_oy - offset_half_f_y]);

                p_input_grads_x[offset_ix] = scalar_t(output_grad_x * cos + output_grad_x_rotate * sin);
                p_input_grads_y[offset_iy] = scalar_t(output_grad_y * cos + output_grad_y_rotate * sin);
            }
        }

        // the rest are just forwarded
        if (size_d > size_f)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ox = hid * stride_ox_h;
                const int32_t offset_oy = hid * stride_oy_h;
                const int32_t offset_ix = hid * stride_ix_h;
                const int32_t offset_iy = hid * stride_iy_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_f; did < size_d; did += blockDim.x)
                {
                    p_input_grads_x[offset_ix + did * stride_ix_d] = p_output_grads_x[offset_ox + did * stride_ox_d];
                    p_input_grads_y[offset_iy + did * stride_iy_d] = p_output_grads_y[offset_oy + did * stride_oy_d];
                }
            }
        }
    }
};

// =====================================================================================================================
// Kernel Entries
//

template <typename Op, typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_1c_sbhd_uncached(
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

    Op{}(
        p_output + offset_o,
        p_input + offset_i,
        p_freqs + offset_f,
        size_h, size_d, size_f,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename Op, typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_uncached(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const int32_t sid = blockIdx.x;
    const int32_t bid = blockIdx.y;
    const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const int32_t offset_f = sid * size_f;

    Op{}(
        p_output_x + offset_ox,
        p_output_y + offset_oy,
        p_input_x + offset_ix,
        p_input_y + offset_iy,
        p_freqs + offset_f,
        size_h, size_d, size_f,
        stride_ix_h, stride_ix_d,
        stride_iy_h, stride_iy_d,
        stride_ox_h, stride_ox_d,
        stride_oy_h, stride_oy_d);
}

template <typename Op, typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_1c_sbhd_cached(
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

    Op{}(
        p_output + offset_o,
        p_input + offset_i,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h, size_d, size_f,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename Op, typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_cached(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const int32_t sid = blockIdx.x;
    const int32_t bid = blockIdx.y;
    const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const int32_t offset_f = sid * size_f;

    Op{}(
        p_output_x + offset_ox,
        p_output_y + offset_oy,
        p_input_x + offset_ix,
        p_input_y + offset_iy,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h, size_d, size_f,
        stride_ix_h, stride_ix_d,
        stride_iy_h, stride_iy_d,
        stride_ox_h, stride_ox_d,
        stride_oy_h, stride_oy_d);
}

template <typename Op, typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_1c_thd_uncached(
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

        Op{}(
            p_output + offset_o,
            p_input + offset_i,
            p_freqs + offset_f,
            size_h, size_d, size_f,
            stride_i_h, stride_i_d,
            stride_o_h, stride_o_d);
    }
}

template <typename Op, typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_1c_2d_cached(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_cos_h,
    const scalar_f_t* __restrict__ p_sin_h,
    const scalar_f_t* __restrict__ p_cos_w,
    const scalar_f_t* __restrict__ p_sin_w,
    const int32_t img_width, const int32_t size_h, const int32_t size_d,
    const int32_t stride_i_b, const int32_t stride_i_s, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_b, const int32_t stride_o_s, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int Hid = blockIdx.x;
    const int Wid = blockIdx.y;
    const int sid = Hid * img_width + Wid;
    const int bid = blockIdx.z;
    const int size_half_d = size_d >> 1;

    const int offset_h_i = bid * stride_i_b + sid * stride_i_s;
    const int offset_h_o = bid * stride_o_b + sid * stride_o_s;
    const int offset_h_f = Hid * size_half_d;
    Op{}(
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
    Op{}(
        p_output + offset_w_o,
        p_input + offset_w_i,
        p_cos_w + offset_w_f,
        p_sin_w + offset_w_f,
        size_h, size_half_d, size_half_d,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

// =====================================================================================================================
// Dispatches
//

template <typename Op, typename scalar_t, typename scalar_f_t>
void dispatch_1c_sbhd_uncached(
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

    kn_entry_1c_sbhd_uncached<Op><<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_freqs,
        size_h, size_d, size_f,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d);
}

template <typename Op, typename scalar_t, typename scalar_f_t>
void dispatch_2c_sbhd_uncached(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_entry_2c_sbhd_uncached<Op><<<grid, block, 0, stream>>>(
        p_output_x,
        p_output_y,
        p_input_x,
        p_input_y,
        p_freqs,
        size_h, size_d, size_f,
        stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
        stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
        stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
        stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
}

template <typename Op, typename scalar_t, typename scalar_f_t>
void dispatch_1c_sbhd_cached(
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

    kn_entry_1c_sbhd_cached<Op><<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_cos, p_sin,
        size_h, size_d, size_f,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d);
}

template <typename Op, typename scalar_t, typename scalar_f_t>
void dispatch_2c_sbhd_cached(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_s, const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_entry_2c_sbhd_cached<Op><<<grid, block, 0, stream>>>(
        p_output_x,
        p_output_y,
        p_input_x,
        p_input_y,
        p_cos, p_sin,
        size_h, size_d, size_f,
        stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
        stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
        stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
        stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
}

template <typename Op, typename scalar_t, typename scalar_f_t>
void dispatch_1c_thd_uncached(
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

    kn_entry_1c_thd_uncached<Op><<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_cu_seqlens,
        p_freqs,
        size_h, size_d, size_f,
        stride_i_t, stride_i_h, stride_i_d,
        stride_o_t, stride_o_h, stride_o_d);
}

template <typename Op, typename scalar_t, typename scalar_f_t>
void dispatch_1c_2d_cached(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_cos_h,
    const scalar_f_t* __restrict__ p_sin_h,
    const scalar_f_t* __restrict__ p_cos_w,
    const scalar_f_t* __restrict__ p_sin_w,
    const int img_height, const int img_width,
    const int32_t size_b, const int32_t size_h, const int32_t size_d,
    const int32_t stride_i_b, const int32_t stride_i_s, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_b, const int32_t stride_o_s, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(img_height, img_width, size_b);
    const dim3 block(C10_WARP_SIZE, size_h < 16 ? 4 : 8);

    kn_entry_1c_2d_cached<Op><<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_cos_h, p_sin_h,
        p_cos_w, p_sin_w,
        img_width, size_h, size_d,
        stride_i_b, stride_i_s, stride_i_h, stride_i_d,
        stride_o_b, stride_o_s, stride_o_h, stride_o_d);
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
                dispatch_1c_sbhd_uncached<Op1cUncachedFwd>(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    freqs.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::Half:
                dispatch_1c_sbhd_uncached<Op1cUncachedFwd>(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_1c_sbhd_uncached<Op1cUncachedFwd>(
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
                dispatch_1c_sbhd_uncached<Op1cUncachedBwd>(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    freqs.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d,
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::Half:
                dispatch_1c_sbhd_uncached<Op1cUncachedBwd>(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_o_s, stride_o_b, stride_o_h, stride_o_d,
                    stride_i_s, stride_i_b, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_1c_sbhd_uncached<Op1cUncachedBwd>(
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

void rope_2c_fwd_impl(
    torch::Tensor&       output_x,      // [s, b, h, d]
    torch::Tensor&       output_y,      // [s, b, h, d]
    const torch::Tensor& input_x,       // [s, b, h, d]
    const torch::Tensor& input_y,       // [s, b, h, d]
    const torch::Tensor& freqs)         // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = input_x.size(0);
    const int32_t size_b = input_x.size(1);
    const int32_t size_h = input_x.size(2);
    const int32_t size_d = input_x.size(3);
    const int32_t size_f = freqs.size(3);
    // Get strides of input
    const int32_t stride_ix_s = input_x.stride(0);
    const int32_t stride_ix_b = input_x.stride(1);
    const int32_t stride_ix_h = input_x.stride(2);
    const int32_t stride_ix_d = input_x.stride(3);
    const int32_t stride_iy_s = input_y.stride(0);
    const int32_t stride_iy_b = input_y.stride(1);
    const int32_t stride_iy_h = input_y.stride(2);
    const int32_t stride_iy_d = input_y.stride(3);
    // Get strides of output
    const int32_t stride_ox_s = output_x.stride(0);
    const int32_t stride_ox_b = output_x.stride(1);
    const int32_t stride_ox_h = output_x.stride(2);
    const int32_t stride_ox_d = output_x.stride(3);
    const int32_t stride_oy_s = output_y.stride(0);
    const int32_t stride_oy_b = output_y.stride(1);
    const int32_t stride_oy_h = output_y.stride(2);
    const int32_t stride_oy_d = output_y.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        input_x.scalar_type(),
        "kn_rope_fwd",
        [&] {
            switch (freqs.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_2c_sbhd_uncached<Op2cUncachedFwd>(
                    output_x.data_ptr<scalar_t>(),
                    output_y.data_ptr<scalar_t>(),
                    input_x.data_ptr<scalar_t>(),
                    input_y.data_ptr<scalar_t>(),
                    freqs.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
                break;
            case at::ScalarType::Half:
                dispatch_2c_sbhd_uncached<Op2cUncachedFwd>(
                    output_x.data_ptr<scalar_t>(),
                    output_y.data_ptr<scalar_t>(),
                    input_x.data_ptr<scalar_t>(),
                    input_y.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_2c_sbhd_uncached<Op2cUncachedFwd>(
                    output_x.data_ptr<scalar_t>(),
                    output_y.data_ptr<scalar_t>(),
                    input_x.data_ptr<scalar_t>(),
                    input_y.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::BFloat16>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_fwd doesn't support to specified formats.");
                break;
            }
        });
}

void rope_2c_bwd_impl(
    torch::Tensor&       input_grads_x, // [s, b, h, d]
    torch::Tensor&       input_grads_y, // [s, b, h, d]
    const torch::Tensor& output_grads_x,// [s, b, h, d]
    const torch::Tensor& output_grads_y,// [s, b, h, d]
    const torch::Tensor& freqs)         // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = output_grads_x.size(0);
    const int32_t size_b = output_grads_x.size(1);
    const int32_t size_h = output_grads_x.size(2);
    const int32_t size_d = output_grads_x.size(3);
    const int32_t size_f = freqs.size(3);
    // Get strides of output_grads
    const int32_t stride_ox_s = output_grads_x.stride(0);
    const int32_t stride_ox_b = output_grads_x.stride(1);
    const int32_t stride_ox_h = output_grads_x.stride(2);
    const int32_t stride_ox_d = output_grads_x.stride(3);
    const int32_t stride_oy_s = output_grads_y.stride(0);
    const int32_t stride_oy_b = output_grads_y.stride(1);
    const int32_t stride_oy_h = output_grads_y.stride(2);
    const int32_t stride_oy_d = output_grads_y.stride(3);
    // Get strides of input_grads
    const int32_t stride_ix_s = input_grads_x.stride(0);
    const int32_t stride_ix_b = input_grads_x.stride(1);
    const int32_t stride_ix_h = input_grads_x.stride(2);
    const int32_t stride_ix_d = input_grads_x.stride(3);
    const int32_t stride_iy_s = input_grads_y.stride(0);
    const int32_t stride_iy_b = input_grads_y.stride(1);
    const int32_t stride_iy_h = input_grads_y.stride(2);
    const int32_t stride_iy_d = input_grads_y.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        output_grads_x.scalar_type(),
        "kn_rope_bwd",
        [&] {
            switch (freqs.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_2c_sbhd_uncached<Op2cUncachedBwd>(
                    input_grads_x.data_ptr<scalar_t>(),
                    input_grads_y.data_ptr<scalar_t>(),
                    output_grads_x.data_ptr<scalar_t>(),
                    output_grads_y.data_ptr<scalar_t>(),
                    freqs.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
                break;
            case at::ScalarType::Half:
                dispatch_2c_sbhd_uncached<Op2cUncachedBwd>(
                    input_grads_x.data_ptr<scalar_t>(),
                    input_grads_y.data_ptr<scalar_t>(),
                    output_grads_x.data_ptr<scalar_t>(),
                    output_grads_y.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_2c_sbhd_uncached<Op2cUncachedBwd>(
                    input_grads_x.data_ptr<scalar_t>(),
                    input_grads_y.data_ptr<scalar_t>(),
                    output_grads_x.data_ptr<scalar_t>(),
                    output_grads_y.data_ptr<scalar_t>(),
                    freqs.data_ptr<at::BFloat16>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
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
                dispatch_1c_sbhd_cached<Op1cCachedFwd>(
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
                dispatch_1c_sbhd_cached<Op1cCachedFwd>(
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
                dispatch_1c_sbhd_cached<Op1cCachedFwd>(
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
                dispatch_1c_sbhd_cached<Op1cCachedBwd>(
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
                dispatch_1c_sbhd_cached<Op1cCachedBwd>(
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
                dispatch_1c_sbhd_cached<Op1cCachedBwd>(
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

void rope_cached_2c_fwd_impl(
    torch::Tensor&       output_x,      // [s, b, h, d]
    torch::Tensor&       output_y,      // [s, b, h, d]
    const torch::Tensor& input_x,       // [s, b, h, d]
    const torch::Tensor& input_y,       // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin)           // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = input_x.size(0);
    const int32_t size_b = input_x.size(1);
    const int32_t size_h = input_x.size(2);
    const int32_t size_d = input_x.size(3);
    const int32_t size_f = cos.size(3);
    // Get strides of input
    const int32_t stride_ix_s = input_x.stride(0);
    const int32_t stride_ix_b = input_x.stride(1);
    const int32_t stride_ix_h = input_x.stride(2);
    const int32_t stride_ix_d = input_x.stride(3);
    const int32_t stride_iy_s = input_y.stride(0);
    const int32_t stride_iy_b = input_y.stride(1);
    const int32_t stride_iy_h = input_y.stride(2);
    const int32_t stride_iy_d = input_y.stride(3);
    // Get strides of output
    const int32_t stride_ox_s = output_x.stride(0);
    const int32_t stride_ox_b = output_x.stride(1);
    const int32_t stride_ox_h = output_x.stride(2);
    const int32_t stride_ox_d = output_x.stride(3);
    const int32_t stride_oy_s = output_y.stride(0);
    const int32_t stride_oy_b = output_y.stride(1);
    const int32_t stride_oy_h = output_y.stride(2);
    const int32_t stride_oy_d = output_y.stride(3);

    VLLM_DISPATCH_FLOATING_TYPES(
        input_x.scalar_type(),
        "kn_rope_cached_fwd",
        [&] {
            switch (cos.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_2c_sbhd_cached<Op2cCachedFwd>(
                    output_x.data_ptr<scalar_t>(),
                    output_y.data_ptr<scalar_t>(),
                    input_x.data_ptr<scalar_t>(),
                    input_y.data_ptr<scalar_t>(),
                    cos.data_ptr<float>(),
                    sin.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
                break;
            case at::ScalarType::Half:
                dispatch_2c_sbhd_cached<Op2cCachedFwd>(
                    output_x.data_ptr<scalar_t>(),
                    output_y.data_ptr<scalar_t>(),
                    input_x.data_ptr<scalar_t>(),
                    input_y.data_ptr<scalar_t>(),
                    cos.data_ptr<at::Half>(),
                    sin.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_2c_sbhd_cached<Op2cCachedFwd>(
                    output_x.data_ptr<scalar_t>(),
                    output_y.data_ptr<scalar_t>(),
                    input_x.data_ptr<scalar_t>(),
                    input_y.data_ptr<scalar_t>(),
                    cos.data_ptr<at::BFloat16>(),
                    sin.data_ptr<at::BFloat16>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_cached_fwd doesn't support to specified formats.");
                break;
            }
        });
}

void rope_cached_2c_bwd_impl(
    torch::Tensor&       input_grads_x, // [s, b, h, d]
    torch::Tensor&       input_grads_y, // [s, b, h, d]
    const torch::Tensor& output_grads_x,// [s, b, h, d]
    const torch::Tensor& output_grads_y,// [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin)           // [s, 1, 1, d]
{
    // Get sizes of input and output
    const int32_t size_s = output_grads_x.size(0);
    const int32_t size_b = output_grads_x.size(1);
    const int32_t size_h = output_grads_x.size(2);
    const int32_t size_d = output_grads_x.size(3);
    const int32_t size_f = cos.size(3);
    // Get strides of output_grads
    const int32_t stride_ox_s = output_grads_x.stride(0);
    const int32_t stride_ox_b = output_grads_x.stride(1);
    const int32_t stride_ox_h = output_grads_x.stride(2);
    const int32_t stride_ox_d = output_grads_x.stride(3);
    const int32_t stride_oy_s = output_grads_y.stride(0);
    const int32_t stride_oy_b = output_grads_y.stride(1);
    const int32_t stride_oy_h = output_grads_y.stride(2);
    const int32_t stride_oy_d = output_grads_y.stride(3);
    // Get strides of input_grads
    const int32_t stride_ix_s = input_grads_x.stride(0);
    const int32_t stride_ix_b = input_grads_x.stride(1);
    const int32_t stride_ix_h = input_grads_x.stride(2);
    const int32_t stride_ix_d = input_grads_x.stride(3);
    const int32_t stride_iy_s = input_grads_y.stride(0);
    const int32_t stride_iy_b = input_grads_y.stride(1);
    const int32_t stride_iy_h = input_grads_y.stride(2);
    const int32_t stride_iy_d = input_grads_y.stride(3);


    VLLM_DISPATCH_FLOATING_TYPES(
        output_grads_x.scalar_type(),
        "kn_rope_cached_bwd",
        [&] {
            switch (cos.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_2c_sbhd_cached<Op2cCachedBwd>(
                    input_grads_x.data_ptr<scalar_t>(),
                    input_grads_y.data_ptr<scalar_t>(),
                    output_grads_x.data_ptr<scalar_t>(),
                    output_grads_y.data_ptr<scalar_t>(),
                    cos.data_ptr<float>(),
                    sin.data_ptr<float>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
                break;
            case at::ScalarType::Half:
                dispatch_2c_sbhd_cached<Op2cCachedBwd>(
                    input_grads_x.data_ptr<scalar_t>(),
                    input_grads_y.data_ptr<scalar_t>(),
                    output_grads_x.data_ptr<scalar_t>(),
                    output_grads_y.data_ptr<scalar_t>(),
                    cos.data_ptr<at::Half>(),
                    sin.data_ptr<at::Half>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_2c_sbhd_cached<Op2cCachedBwd>(
                    input_grads_x.data_ptr<scalar_t>(),
                    input_grads_y.data_ptr<scalar_t>(),
                    output_grads_x.data_ptr<scalar_t>(),
                    output_grads_y.data_ptr<scalar_t>(),
                    cos.data_ptr<at::BFloat16>(),
                    sin.data_ptr<at::BFloat16>(),
                    size_s, size_b, size_h, size_d,
                    size_f, // size of last dimension of freqs.
                    stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                    stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
                    stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                    stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
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
                dispatch_1c_thd_uncached<Op1cUncachedFwd>(
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
                dispatch_1c_thd_uncached<Op1cUncachedFwd>(
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
                dispatch_1c_thd_uncached<Op1cUncachedFwd>(
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
                dispatch_1c_thd_uncached<Op1cUncachedBwd>(
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
                dispatch_1c_thd_uncached<Op1cUncachedBwd>(
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
                dispatch_1c_thd_uncached<Op1cUncachedBwd>(
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
    const torch::Tensor& sin_w,
    const int            img_height,
    const int            img_width)
{
    // Get sizes of input and output
    const int size_b = input.size(0);
    const int size_s = input.size(1);
    const int size_h = input.size(2);
    const int size_d = input.size(3);
    // Get strides of input
    const int stride_i_b = input.stride(0);
    const int stride_i_s = input.stride(1);
    const int stride_i_h = input.stride(2);
    const int stride_i_d = input.stride(3);
    // Get strides of output
    const int stride_o_b = output.stride(0);
    const int stride_o_s = output.stride(1);
    const int stride_o_h = output.stride(2);
    const int stride_o_d = output.stride(3);

    TORCH_CHECK(size_s == img_height * img_width, "rope_2d_fwd_impl - input tensor shape doesn't match image size.");

    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "kn_rope_2d_fwd",
        [&] {
            switch (cos_h.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_1c_2d_cached<Op1cCachedFwd>(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cos_h.data_ptr<float>(),
                    sin_h.data_ptr<float>(),
                    cos_w.data_ptr<float>(),
                    sin_w.data_ptr<float>(),
                    img_height, img_width,
                    size_b, size_h, size_d,
                    stride_i_b, stride_i_s, stride_i_h, stride_i_d,
                    stride_o_b, stride_o_s, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::Half:
                dispatch_1c_2d_cached<Op1cCachedFwd>(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cos_h.data_ptr<at::Half>(),
                    sin_h.data_ptr<at::Half>(),
                    cos_w.data_ptr<at::Half>(),
                    sin_w.data_ptr<at::Half>(),
                    img_height, img_width,
                    size_b, size_h, size_d,
                    stride_i_b, stride_i_s, stride_i_h, stride_i_d,
                    stride_o_b, stride_o_s, stride_o_h, stride_o_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_1c_2d_cached<Op1cCachedFwd>(
                    output.data_ptr<scalar_t>(),
                    input.data_ptr<scalar_t>(),
                    cos_h.data_ptr<at::BFloat16>(),
                    sin_h.data_ptr<at::BFloat16>(),
                    cos_w.data_ptr<at::BFloat16>(),
                    sin_w.data_ptr<at::BFloat16>(),
                    img_height, img_width,
                    size_b, size_h, size_d,
                    stride_i_b, stride_i_s, stride_i_h, stride_i_d,
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
    const torch::Tensor& sin_w,
    const int            img_height,
    const int            img_width)
{
    // Get sizes of input and output
    const int size_b = output_grads.size(0);
    const int size_s = output_grads.size(1);
    const int size_h = output_grads.size(2);
    const int size_d = output_grads.size(3);
    // Get strides of output_grads
    const int stride_o_b = output_grads.stride(0);
    const int stride_o_s = output_grads.stride(1);
    const int stride_o_h = output_grads.stride(2);
    const int stride_o_d = output_grads.stride(3);
    // Get strides of input_grads
    const int stride_i_b = input_grads.stride(0);
    const int stride_i_s = input_grads.stride(1);
    const int stride_i_h = input_grads.stride(2);
    const int stride_i_d = input_grads.stride(3);

    TORCH_CHECK(size_s == img_height * img_width, "rope_2d_fwd_impl - input tensor shape doesn't match image size.");

    VLLM_DISPATCH_FLOATING_TYPES(
        output_grads.scalar_type(),
        "kn_rope_2d_bwd",
        [&] {
            switch (cos_h.scalar_type())
            {
            case at::ScalarType::Float:
                dispatch_1c_2d_cached<Op1cCachedBwd>(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cos_h.data_ptr<float>(),
                    sin_h.data_ptr<float>(),
                    cos_w.data_ptr<float>(),
                    sin_w.data_ptr<float>(),
                    img_height, img_width,
                    size_b, size_h, size_d,
                    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
                    stride_i_b, stride_i_s, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::Half:
                dispatch_1c_2d_cached<Op1cCachedBwd>(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cos_h.data_ptr<at::Half>(),
                    sin_h.data_ptr<at::Half>(),
                    cos_w.data_ptr<at::Half>(),
                    sin_w.data_ptr<at::Half>(),
                    img_height, img_width,
                    size_b, size_h, size_d,
                    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
                    stride_i_b, stride_i_s, stride_i_h, stride_i_d);
                break;
            case at::ScalarType::BFloat16:
                dispatch_1c_2d_cached<Op1cCachedBwd>(
                    input_grads.data_ptr<scalar_t>(),
                    output_grads.data_ptr<scalar_t>(),
                    cos_h.data_ptr<at::BFloat16>(),
                    sin_h.data_ptr<at::BFloat16>(),
                    cos_w.data_ptr<at::BFloat16>(),
                    sin_w.data_ptr<at::BFloat16>(),
                    img_height, img_width,
                    size_b, size_h, size_d,
                    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
                    stride_i_b, stride_i_s, stride_i_h, stride_i_d);
                break;
            default:
                TORCH_CHECK(false, "kn_rope_2d_bwd doesn't support to specified formats.");
                break;
            }
        });
}
