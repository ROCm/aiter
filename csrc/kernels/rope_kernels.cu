// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <c10/cuda/CUDAGuard.h>
#include "rope.h"
#include "dispatch_utils.h"

// =====================================================================================================================
// Keyword interpretation
// ----------------------------------------------------------------
// 1c/2c:               The number of channels. 2c means two inputs and two outputs.
// Cached, Uncached:    Whether cosine and sine are calculated in kernel. Cached means kernel can read these value from
//                      memory rather than calculate these value according to the given theta in memory.
// ReuseFreqsFrontPart: Normally, freqs/cos/sin tensors should be repeated before conduct the RoPE operators. With
//                      this value set as true, the repeat is no longer required. Kernel can automatically relocate the
//                      desired element.
// sbhd:                Shape of tensor: [sequence length, batch size, head count, hidden dimension].
// thd:                 Shape of tensor.
// 2d:                  2D image.
//

#define ROTATE_STYLE_NEOX 0
#define ROTATE_STYLE_GPTJ 1

// =====================================================================================================================
// Kernel Helper Functions
//

template <int32_t RotateStyle, bool IsForward, bool ReuseFreqsFrontPart, typename scalar_f_t>
inline __device__ void get_cos_sin_uncached(
    float* p_cos_0, float* p_sin_0,
    float* p_cos_1, float* p_sin_1,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t did,
    const int32_t size_half_r)
{
    if constexpr (RotateStyle == ROTATE_STYLE_NEOX)
    {
        if constexpr (IsForward == true)
        {
            sincosf(float(p_freqs[did]), p_sin_0, p_cos_0);
            if constexpr (ReuseFreqsFrontPart)
            {
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                sincosf(float(p_freqs[did + size_half_r]), p_sin_1, p_cos_1);
            }
        }
        else
        {
            const float f_did = float(p_freqs[did]);
            if constexpr (ReuseFreqsFrontPart)
            {
                *p_cos_0 = cosf(f_did);
                *p_sin_0 = sinf(f_did);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                const float f_did_a = p_freqs[did + size_half_r];
                *p_cos_0 = cosf(f_did);
                *p_sin_0 = sinf(f_did_a);
                *p_cos_1 = cosf(f_did_a);
                *p_sin_1 = sinf(f_did);
            }
        }
    }
    else if constexpr (RotateStyle == ROTATE_STYLE_GPTJ)
    {
        if constexpr (IsForward == true)
        {
            if constexpr (ReuseFreqsFrontPart)
            {
                sincosf(float(p_freqs[did]), p_sin_0, p_cos_0);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                sincosf(float(p_freqs[did * 2]),     p_sin_0, p_cos_0);
                sincosf(float(p_freqs[did * 2 + 1]), p_sin_1, p_cos_1);
            }
        }
        else
        {
            if constexpr (ReuseFreqsFrontPart)
            {
                // TODO
            }
            else
            {
                // TODO
            }
        }
    }
}

template <int32_t RotateStyle, bool IsForward, bool ReuseFreqsFrontPart, typename scalar_f_t>
inline __device__ void get_cos_sin_cached(
    float* p_cos_0, float* p_sin_0,
    float* p_cos_1, float* p_sin_1,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t did,
    const int32_t size_half_r)
{
    if constexpr (RotateStyle == ROTATE_STYLE_NEOX)
    {
        if constexpr (IsForward == true)
        {
            *p_cos_0 = float(p_cos[did]);
            *p_sin_0 = float(p_sin[did]);
            if constexpr (ReuseFreqsFrontPart)
            {
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                *p_cos_1 = float(p_cos[did + size_half_r]);
                *p_sin_1 = float(p_sin[did + size_half_r]);
            }
        }
        else
        {
            if constexpr (ReuseFreqsFrontPart)
            {
                *p_cos_0 = float(p_cos[did]);
                *p_sin_0 = float(p_sin[did]);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                *p_cos_0 = float(p_cos[did]);
                *p_sin_0 = float(p_sin[did + size_half_r]);
                *p_cos_1 = float(p_cos[did + size_half_r]);
                *p_sin_1 = float(p_sin[did]);
            }
        }
    }
    else if constexpr (RotateStyle == ROTATE_STYLE_GPTJ)
    {
        if constexpr (IsForward == true)
        {
            if constexpr (ReuseFreqsFrontPart)
            {
                *p_cos_0 = float(p_cos[did]);
                *p_sin_0 = float(p_sin[did]);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                *p_cos_0 = float(p_cos[did * 2]);
                *p_sin_0 = float(p_sin[did * 2]);
                *p_cos_1 = float(p_cos[did * 2 + 1]);
                *p_sin_1 = float(p_sin[did * 2 + 1]);
            }
        }
        else
        {
            if constexpr (ReuseFreqsFrontPart)
            {
                // TODO
            }
            else
            {
                // TODO
            }
        }
    }
}

template <int32_t RotateStyle>
inline __device__ void get_offset_d(
    int32_t* p_offset_0, int32_t* p_offset_1,
    const int32_t did,
    const int32_t stride_d,
    const int32_t offset_half_r) // = stride_d * size_r / 2, size_r = rotate size
{
    if constexpr (RotateStyle == ROTATE_STYLE_NEOX)
    {
        *p_offset_0 = did * stride_d;
        *p_offset_1 = *p_offset_0 + offset_half_r;
    }
    else if constexpr (RotateStyle == ROTATE_STYLE_GPTJ)
    {
        *p_offset_0 = 2 * did * stride_d;
        *p_offset_1 = *p_offset_0 + stride_d;
    }
}

// =====================================================================================================================
// Kernel Functionalities
//

struct Op1cUncachedFwd
{
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
    __device__ static void apply(
        scalar_t* __restrict__         p_output,
        const scalar_t* __restrict__   p_input,
        const scalar_f_t* __restrict__ p_freqs,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_i_h, const int32_t stride_i_d,
        const int32_t stride_o_h, const int32_t stride_o_d)
    {
        // rotate count
        const int32_t size_r        = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r   = size_r >> 1;
        const int32_t offset_half_r_i = size_half_r * stride_i_d;
        const int32_t offset_half_r_o = size_half_r * stride_o_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_half_r; did += blockDim.x)
        {
            // i: Input, o: output, d: in hidden Dim, 0: former element, 1: latter element
            int32_t offset_i_d_0, offset_i_d_1, offset_o_d_0, offset_o_d_1;
            get_offset_d<RotateStyle>(&offset_i_d_0, &offset_i_d_1, did, stride_i_d, offset_half_r_i);
            get_offset_d<RotateStyle>(&offset_o_d_0, &offset_o_d_1, did, stride_o_d, offset_half_r_o);

            float cos_0, sin_0, cos_1, sin_1;
            get_cos_sin_uncached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_i_h = hid * stride_i_h;
                const int32_t offset_o_h = hid * stride_o_h;
                const int32_t offset_i_0 = offset_i_h + offset_i_d_0;
                const int32_t offset_i_1 = offset_i_h + offset_i_d_1;
                const int32_t offset_o_0 = offset_o_h + offset_o_d_0;
                const int32_t offset_o_1 = offset_o_h + offset_o_d_1;

                const float input_0 = float(p_input[offset_i_0]);
                const float input_1 = float(p_input[offset_i_1]);

                p_output[offset_o_0] = scalar_t(input_0 * cos_0 - input_1 * sin_0);
                p_output[offset_o_1] = scalar_t(input_1 * cos_1 + input_0 * sin_1);
            }
        }

        // the rest are just forwarded
        if (size_d > size_r)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_i = hid * stride_i_h;
                const int32_t offset_o = hid * stride_o_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_r; did < size_d; did += blockDim.x)
                {
                    p_output[offset_o + did * stride_o_d] = p_input[offset_i + did * stride_i_d];
                }
            }
        }
    }
};

struct Op1cUncachedBwd
{
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
    __device__ static void apply(
        scalar_t* __restrict__         p_input_grads,
        const scalar_t* __restrict__   p_output_grads,
        const scalar_f_t* __restrict__ p_freqs,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_o_h, const int32_t stride_o_d,
        const int32_t stride_i_h, const int32_t stride_i_d)
    {
        // rotate count
        const int32_t size_r          = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r     = size_r >> 1;
        const int32_t offset_half_r_o = size_half_r * stride_o_d;
        const int32_t offset_half_r_i = size_half_r * stride_i_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_half_r; did += blockDim.x)
        {
            // i: Input grads, o: output grads, d: in hidden Dim, 0: former element, 1: latter element
            int32_t offset_o_d_0, offset_o_d_1, offset_i_d_0, offset_i_d_1;
            get_offset_d<RotateStyle>(&offset_o_d_0, &offset_o_d_1, did, stride_o_d, offset_half_r_o);
            get_offset_d<RotateStyle>(&offset_i_d_0, &offset_i_d_1, did, stride_i_d, offset_half_r_i);

            float cos_0, sin_0, cos_1, sin_1;
            get_cos_sin_uncached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_o_h = hid * stride_o_h;
                const int32_t offset_i_h = hid * stride_i_h;
                const int32_t offset_o_0 = offset_o_h + offset_o_d_0;
                const int32_t offset_o_1 = offset_o_h + offset_o_d_1;
                const int32_t offset_i_0 = offset_i_h + offset_i_d_0;
                const int32_t offset_i_1 = offset_i_h + offset_i_d_1;

                const float output_grad_0 = float(p_output_grads[offset_o_0]);
                const float output_grad_1 = float(p_output_grads[offset_o_1]);

                p_input_grads[offset_i_0] = scalar_t(output_grad_0 * cos_0 + output_grad_1 * sin_0);
                p_input_grads[offset_i_1] = scalar_t(output_grad_1 * cos_1 - output_grad_0 * sin_1);
            }
        }

        // the rest are just forwarded
        if (size_d > size_r)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_o = hid * stride_o_h;
                const int32_t offset_i = hid * stride_i_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_r; did < size_d; did += blockDim.x)
                {
                    p_input_grads[offset_i + did * stride_i_d] = p_output_grads[offset_o + did * stride_o_d];
                }
            }
        }
    }
};

struct Op2cUncachedFwd
{
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
    __device__ static void apply(
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
        // rotate count
        const int32_t size_r           = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r      = size_r >> 1;
        const int32_t offset_half_r_ix = size_half_r * stride_ix_d;
        const int32_t offset_half_r_ox = size_half_r * stride_ox_d;
        const int32_t offset_half_r_iy = size_half_r * stride_iy_d;
        const int32_t offset_half_r_oy = size_half_r * stride_oy_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_half_r; did += blockDim.x)
        {
            // i: Input, o: output, d: in hidden Dim, 0: former element, 1: latter element, x: 1st channel, y: 2nd channel
            int32_t offset_ix_d_0, offset_ix_d_1, offset_ox_d_0, offset_ox_d_1;
            get_offset_d<RotateStyle>(&offset_ix_d_0, &offset_ix_d_1, did, stride_ix_d, offset_half_r_ix);
            get_offset_d<RotateStyle>(&offset_ox_d_0, &offset_ox_d_1, did, stride_ox_d, offset_half_r_ox);
            int32_t offset_iy_d_0, offset_iy_d_1, offset_oy_d_0, offset_oy_d_1;
            get_offset_d<RotateStyle>(&offset_iy_d_0, &offset_iy_d_1, did, stride_iy_d, offset_half_r_iy);
            get_offset_d<RotateStyle>(&offset_oy_d_0, &offset_oy_d_1, did, stride_oy_d, offset_half_r_oy);

            float cos_0, sin_0, cos_1, sin_1;
            get_cos_sin_uncached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ix_h = hid * stride_ix_h;
                const int32_t offset_ox_h = hid * stride_ox_h;
                const int32_t offset_iy_h = hid * stride_iy_h;
                const int32_t offset_oy_h = hid * stride_oy_h;
                const int32_t offset_ix_0 = offset_ix_h + offset_ix_d_0;
                const int32_t offset_ix_1 = offset_ix_h + offset_ix_d_1;
                const int32_t offset_ox_0 = offset_ox_h + offset_ox_d_0;
                const int32_t offset_ox_1 = offset_ox_h + offset_ox_d_1;
                const int32_t offset_iy_0 = offset_iy_h + offset_iy_d_0;
                const int32_t offset_iy_1 = offset_iy_h + offset_iy_d_1;
                const int32_t offset_oy_0 = offset_oy_h + offset_oy_d_0;
                const int32_t offset_oy_1 = offset_oy_h + offset_oy_d_1;

                const float input_x_0 = float(p_input_x[offset_ix_0]);
                const float input_x_1 = float(p_input_x[offset_ix_1]);
                const float input_y_0 = float(p_input_y[offset_iy_0]);
                const float input_y_1 = float(p_input_y[offset_iy_1]);

                p_output_x[offset_ox_0] = scalar_t(input_x_0 * cos_0 - input_x_1 * sin_0);
                p_output_x[offset_ox_1] = scalar_t(input_x_1 * cos_1 + input_x_0 * sin_1);
                p_output_y[offset_oy_0] = scalar_t(input_y_0 * cos_0 - input_y_1 * sin_0);
                p_output_y[offset_oy_1] = scalar_t(input_y_1 * cos_1 + input_y_0 * sin_1);
            }
        }

        // the rest are just forwarded
        if (size_d > size_r)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ix = hid * stride_ix_h;
                const int32_t offset_iy = hid * stride_iy_h;
                const int32_t offset_ox = hid * stride_ox_h;
                const int32_t offset_oy = hid * stride_oy_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_r; did < size_d; did += blockDim.x)
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
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
    __device__ static void apply(
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
        // rotate count
        const int32_t size_r           = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r      = size_r >> 1;
        const int32_t offset_half_r_ox = size_half_r * stride_ox_d;
        const int32_t offset_half_r_ix = size_half_r * stride_ix_d;
        const int32_t offset_half_r_oy = size_half_r * stride_oy_d;
        const int32_t offset_half_r_iy = size_half_r * stride_iy_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_half_r; did += blockDim.x)
        {
            // i: Input, o: output, d: in hidden Dim, 0: former element, 1: latter element, x: 1st channel, y: 2nd channel
            int32_t offset_ox_d_0, offset_ox_d_1, offset_ix_d_0, offset_ix_d_1;
            get_offset_d<RotateStyle>(&offset_ox_d_0, &offset_ox_d_1, did, stride_ox_d, offset_half_r_ox);
            get_offset_d<RotateStyle>(&offset_ix_d_0, &offset_ix_d_1, did, stride_ix_d, offset_half_r_ix);
            int32_t offset_oy_d_0, offset_oy_d_1, offset_iy_d_0, offset_iy_d_1;
            get_offset_d<RotateStyle>(&offset_oy_d_0, &offset_oy_d_1, did, stride_oy_d, offset_half_r_oy);
            get_offset_d<RotateStyle>(&offset_iy_d_0, &offset_iy_d_1, did, stride_iy_d, offset_half_r_iy);

            float cos_0, sin_0, cos_1, sin_1;
            get_cos_sin_uncached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ox_h = hid * stride_ox_h;
                const int32_t offset_ix_h = hid * stride_ix_h;
                const int32_t offset_oy_h = hid * stride_oy_h;
                const int32_t offset_iy_h = hid * stride_iy_h;
                const int32_t offset_ox_0 = offset_ox_h + offset_ox_d_0;
                const int32_t offset_ox_1 = offset_ox_h + offset_ox_d_1;
                const int32_t offset_ix_0 = offset_ix_h + offset_ix_d_0;
                const int32_t offset_ix_1 = offset_ix_h + offset_ix_d_1;
                const int32_t offset_oy_0 = offset_oy_h + offset_oy_d_0;
                const int32_t offset_oy_1 = offset_oy_h + offset_oy_d_1;
                const int32_t offset_iy_0 = offset_iy_h + offset_iy_d_0;
                const int32_t offset_iy_1 = offset_iy_h + offset_iy_d_1;

                const float output_grad_x_0 = float(p_output_grads_x[offset_ox_0]);
                const float output_grad_x_1 = float(p_output_grads_x[offset_ox_1]);
                const float output_grad_y_0 = float(p_output_grads_y[offset_oy_0]);
                const float output_grad_y_1 = float(p_output_grads_y[offset_oy_1]);

                p_input_grads_x[offset_ix_0] = scalar_t(output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0);
                p_input_grads_x[offset_ix_1] = scalar_t(output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1);
                p_input_grads_y[offset_iy_0] = scalar_t(output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0);
                p_input_grads_y[offset_iy_1] = scalar_t(output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1);
            }
        }

        // the rest are just forwarded
        if (size_d > size_r)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ox = hid * stride_ox_h;
                const int32_t offset_oy = hid * stride_oy_h;
                const int32_t offset_ix = hid * stride_ix_h;
                const int32_t offset_iy = hid * stride_iy_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_r; did < size_d; did += blockDim.x)
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
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
    __device__ static void apply(
        scalar_t* __restrict__         p_output,
        const scalar_t* __restrict__   p_input,
        const scalar_f_t* __restrict__ p_cos,
        const scalar_f_t* __restrict__ p_sin,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_i_h, const int32_t stride_i_d,
        const int32_t stride_o_h, const int32_t stride_o_d)
    {
        // rotate count
        const int32_t size_r        = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r   = size_r >> 1;
        const int32_t offset_half_r_i = size_half_r * stride_i_d;
        const int32_t offset_half_r_o = size_half_r * stride_o_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_half_r; did += blockDim.x)
        {
            // i: Input, o: output, d: in hidden Dim, 0: former element, 1: latter element
            int32_t offset_i_d_0, offset_i_d_1, offset_o_d_0, offset_o_d_1;
            get_offset_d<RotateStyle>(&offset_i_d_0, &offset_i_d_1, did, stride_i_d, offset_half_r_i);
            get_offset_d<RotateStyle>(&offset_o_d_0, &offset_o_d_1, did, stride_o_d, offset_half_r_o);

            float cos_0, sin_0, cos_1, sin_1;
            get_cos_sin_cached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_i_h = hid * stride_i_h;
                const int32_t offset_o_h = hid * stride_o_h;
                const int32_t offset_i_0 = offset_i_h + offset_i_d_0;
                const int32_t offset_i_1 = offset_i_h + offset_i_d_1;
                const int32_t offset_o_0 = offset_o_h + offset_o_d_0;
                const int32_t offset_o_1 = offset_o_h + offset_o_d_1;

                const float input_0 = float(p_input[offset_i_0]);
                const float input_1 = float(p_input[offset_i_1]);

                p_output[offset_o_0] = scalar_t(input_0 * cos_0 - input_1 * sin_0);
                p_output[offset_o_1] = scalar_t(input_1 * cos_1 + input_0 * sin_1);
            }
        }

        // the rest are just forwarded
        if (size_d > size_r)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_i = hid * stride_i_h;
                const int32_t offset_o = hid * stride_o_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_r; did < size_d; did += blockDim.x)
                {
                    p_output[offset_o + did * stride_o_d] = p_input[offset_i + did * stride_i_d];
                }
            }
        }
    }
};

struct Op1cCachedBwd
{
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
    __device__ static void apply(
        scalar_t* __restrict__         p_input_grads,
        const scalar_t* __restrict__   p_output_grads,
        const scalar_f_t* __restrict__ p_cos,
        const scalar_f_t* __restrict__ p_sin,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_o_h, const int32_t stride_o_d,
        const int32_t stride_i_h, const int32_t stride_i_d)
    {
        // rotate count
        const int32_t size_r          = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r     = size_r >> 1;
        const int32_t offset_half_r_o = size_half_r * stride_o_d;
        const int32_t offset_half_r_i = size_half_r * stride_i_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_half_r; did += blockDim.x)
        {
            // i: Input grads, o: output grads, d: in hidden Dim, 0: former element, 1: latter element
            int32_t offset_o_d_0, offset_o_d_1, offset_i_d_0, offset_i_d_1;
            get_offset_d<RotateStyle>(&offset_o_d_0, &offset_o_d_1, did, stride_o_d, offset_half_r_o);
            get_offset_d<RotateStyle>(&offset_i_d_0, &offset_i_d_1, did, stride_i_d, offset_half_r_i);

            float cos_0, sin_0, cos_1, sin_1;
            get_cos_sin_cached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_o_h = hid * stride_o_h;
                const int32_t offset_i_h = hid * stride_i_h;
                const int32_t offset_o_0 = offset_o_h + offset_o_d_0;
                const int32_t offset_o_1 = offset_o_h + offset_o_d_1;
                const int32_t offset_i_0 = offset_i_h + offset_i_d_0;
                const int32_t offset_i_1 = offset_i_h + offset_i_d_1;

                const float output_grad_0 = float(p_output_grads[offset_o_0]);
                const float output_grad_1 = float(p_output_grads[offset_o_1]);

                p_input_grads[offset_i_0] = scalar_t(output_grad_0 * cos_0 + output_grad_1 * sin_0);
                p_input_grads[offset_i_1] = scalar_t(output_grad_1 * cos_1 - output_grad_0 * sin_1);
            }
        }

        // the rest are just forwarded
        if (size_d > size_r)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_o = hid * stride_o_h;
                const int32_t offset_i = hid * stride_i_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_r; did < size_d; did += blockDim.x)
                {
                    p_input_grads[offset_i + did * stride_i_d] = p_output_grads[offset_o + did * stride_o_d];
                }
            }
        }
    }
};

struct Op2cCachedFwd
{
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
    __device__ static void apply(
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
        // rotate count
        const int32_t size_r           = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r      = size_r >> 1;
        const int32_t offset_half_r_ix = size_half_r * stride_ix_d;
        const int32_t offset_half_r_ox = size_half_r * stride_ox_d;
        const int32_t offset_half_r_iy = size_half_r * stride_iy_d;
        const int32_t offset_half_r_oy = size_half_r * stride_oy_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_half_r; did += blockDim.x)
        {
            // i: Input, o: output, d: in hidden Dim, 0: former element, 1: latter element, x: 1st channel, y: 2nd channel
            int32_t offset_ix_d_0, offset_ix_d_1, offset_ox_d_0, offset_ox_d_1;
            get_offset_d<RotateStyle>(&offset_ix_d_0, &offset_ix_d_1, did, stride_ix_d, offset_half_r_ix);
            get_offset_d<RotateStyle>(&offset_ox_d_0, &offset_ox_d_1, did, stride_ox_d, offset_half_r_ox);
            int32_t offset_iy_d_0, offset_iy_d_1, offset_oy_d_0, offset_oy_d_1;
            get_offset_d<RotateStyle>(&offset_iy_d_0, &offset_iy_d_1, did, stride_iy_d, offset_half_r_iy);
            get_offset_d<RotateStyle>(&offset_oy_d_0, &offset_oy_d_1, did, stride_oy_d, offset_half_r_oy);

            float cos_0, sin_0, cos_1, sin_1;
            get_cos_sin_cached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ix_h = hid * stride_ix_h;
                const int32_t offset_ox_h = hid * stride_ox_h;
                const int32_t offset_iy_h = hid * stride_iy_h;
                const int32_t offset_oy_h = hid * stride_oy_h;
                const int32_t offset_ix_0 = offset_ix_h + offset_ix_d_0;
                const int32_t offset_ix_1 = offset_ix_h + offset_ix_d_1;
                const int32_t offset_ox_0 = offset_ox_h + offset_ox_d_0;
                const int32_t offset_ox_1 = offset_ox_h + offset_ox_d_1;
                const int32_t offset_iy_0 = offset_iy_h + offset_iy_d_0;
                const int32_t offset_iy_1 = offset_iy_h + offset_iy_d_1;
                const int32_t offset_oy_0 = offset_oy_h + offset_oy_d_0;
                const int32_t offset_oy_1 = offset_oy_h + offset_oy_d_1;

                const float input_x_0 = float(p_input_x[offset_ix_0]);
                const float input_x_1 = float(p_input_x[offset_ix_1]);
                const float input_y_0 = float(p_input_y[offset_iy_0]);
                const float input_y_1 = float(p_input_y[offset_iy_1]);

                p_output_x[offset_ox_0] = scalar_t(input_x_0 * cos_0 - input_x_1 * sin_0);
                p_output_x[offset_ox_1] = scalar_t(input_x_1 * cos_1 + input_x_0 * sin_1);
                p_output_y[offset_oy_0] = scalar_t(input_y_0 * cos_0 - input_y_1 * sin_0);
                p_output_y[offset_oy_1] = scalar_t(input_y_1 * cos_1 + input_y_0 * sin_1);
            }
        }

        // the rest are just forwarded
        if (size_d > size_r)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ix = hid * stride_ix_h;
                const int32_t offset_iy = hid * stride_iy_h;
                const int32_t offset_ox = hid * stride_ox_h;
                const int32_t offset_oy = hid * stride_oy_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_r; did < size_d; did += blockDim.x)
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
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
    __device__ static void apply(
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
        // rotate count
        const int32_t size_r           = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r      = size_r >> 1;
        const int32_t offset_half_r_ox = size_half_r * stride_ox_d;
        const int32_t offset_half_r_ix = size_half_r * stride_ix_d;
        const int32_t offset_half_r_oy = size_half_r * stride_oy_d;
        const int32_t offset_half_r_iy = size_half_r * stride_iy_d;

        #pragma unroll
        for (int32_t did = threadIdx.x; did < size_half_r; did += blockDim.x)
        {
            // i: Input, o: output, d: in hidden Dim, 0: former element, 1: latter element, x: 1st channel, y: 2nd channel
            int32_t offset_ox_d_0, offset_ox_d_1, offset_ix_d_0, offset_ix_d_1;
            get_offset_d<RotateStyle>(&offset_ox_d_0, &offset_ox_d_1, did, stride_ox_d, offset_half_r_ox);
            get_offset_d<RotateStyle>(&offset_ix_d_0, &offset_ix_d_1, did, stride_ix_d, offset_half_r_ix);
            int32_t offset_oy_d_0, offset_oy_d_1, offset_iy_d_0, offset_iy_d_1;
            get_offset_d<RotateStyle>(&offset_oy_d_0, &offset_oy_d_1, did, stride_oy_d, offset_half_r_oy);
            get_offset_d<RotateStyle>(&offset_iy_d_0, &offset_iy_d_1, did, stride_iy_d, offset_half_r_iy);

            float cos_0, sin_0, cos_1, sin_1;
            get_cos_sin_cached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ox_h = hid * stride_ox_h;
                const int32_t offset_ix_h = hid * stride_ix_h;
                const int32_t offset_oy_h = hid * stride_oy_h;
                const int32_t offset_iy_h = hid * stride_iy_h;
                const int32_t offset_ox_0 = offset_ox_h + offset_ox_d_0;
                const int32_t offset_ox_1 = offset_ox_h + offset_ox_d_1;
                const int32_t offset_ix_0 = offset_ix_h + offset_ix_d_0;
                const int32_t offset_ix_1 = offset_ix_h + offset_ix_d_1;
                const int32_t offset_oy_0 = offset_oy_h + offset_oy_d_0;
                const int32_t offset_oy_1 = offset_oy_h + offset_oy_d_1;
                const int32_t offset_iy_0 = offset_iy_h + offset_iy_d_0;
                const int32_t offset_iy_1 = offset_iy_h + offset_iy_d_1;

                const float output_grad_x_0 = float(p_output_grads_x[offset_ox_0]);
                const float output_grad_x_1 = float(p_output_grads_x[offset_ox_1]);
                const float output_grad_y_0 = float(p_output_grads_y[offset_oy_0]);
                const float output_grad_y_1 = float(p_output_grads_y[offset_oy_1]);

                p_input_grads_x[offset_ix_0] = scalar_t(output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0);
                p_input_grads_x[offset_ix_1] = scalar_t(output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1);
                p_input_grads_y[offset_iy_0] = scalar_t(output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0);
                p_input_grads_y[offset_iy_1] = scalar_t(output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1);
            }
        }

        // the rest are just forwarded
        if (size_d > size_r)
        {
            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                const int32_t offset_ox = hid * stride_ox_h;
                const int32_t offset_oy = hid * stride_oy_h;
                const int32_t offset_ix = hid * stride_ix_h;
                const int32_t offset_iy = hid * stride_iy_h;

                #pragma unroll
                for (int32_t did = threadIdx.x + size_r; did < size_d; did += blockDim.x)
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

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    Op::template apply<RotateStyle, ReuseFreqsFrontPart>(
        p_output + offset_o,
        p_input + offset_i,
        p_freqs + offset_f,
        size_h, size_d, size_f,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    Op::template apply<RotateStyle, ReuseFreqsFrontPart>(
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

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    Op::template apply<RotateStyle, ReuseFreqsFrontPart>(
        p_output + offset_o,
        p_input + offset_i,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h, size_d, size_f,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    Op::template apply<RotateStyle, ReuseFreqsFrontPart>(
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

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

        Op::template apply<RotateStyle, ReuseFreqsFrontPart>(
            p_output + offset_o,
            p_input + offset_i,
            p_freqs + offset_f,
            size_h, size_d, size_f,
            stride_i_h, stride_i_d,
            stride_o_h, stride_o_d);
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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
    Op::template apply<RotateStyle, ReuseFreqsFrontPart>(
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
    Op::template apply<RotateStyle, ReuseFreqsFrontPart>(
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

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    kn_entry_1c_sbhd_uncached<Op, RotateStyle, ReuseFreqsFrontPart><<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_freqs,
        size_h, size_d, size_f,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    kn_entry_2c_sbhd_uncached<Op, RotateStyle, ReuseFreqsFrontPart><<<grid, block, 0, stream>>>(
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

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    kn_entry_1c_sbhd_cached<Op, RotateStyle, ReuseFreqsFrontPart><<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_cos, p_sin,
        size_h, size_d, size_f,
        stride_i_s, stride_i_b, stride_i_h, stride_i_d,
        stride_o_s, stride_o_b, stride_o_h, stride_o_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    kn_entry_2c_sbhd_cached<Op, RotateStyle, ReuseFreqsFrontPart><<<grid, block, 0, stream>>>(
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

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    kn_entry_1c_thd_uncached<Op, RotateStyle, ReuseFreqsFrontPart><<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_cu_seqlens,
        p_freqs,
        size_h, size_d, size_f,
        stride_i_t, stride_i_h, stride_i_d,
        stride_o_t, stride_o_h, stride_o_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, typename scalar_t, typename scalar_f_t>
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

    kn_entry_1c_2d_cached<Op, RotateStyle, ReuseFreqsFrontPart><<<grid, block, 0, stream>>>(
        p_output,
        p_input,
        p_cos_h, p_sin_h,
        p_cos_w, p_sin_w,
        img_width, size_h, size_d,
        stride_i_b, stride_i_s, stride_i_h, stride_i_d,
        stride_o_b, stride_o_s, stride_o_h, stride_o_d);
}

#define DISPATCH_ROPE_TYPES_PARAMS(TYPE0, TYPE1, ROTATE_STYLE, REUSE_FREQS_FRONT_PART, NAME, ...)      \
    switch(TYPE0) {                                                                                    \
        case at::ScalarType::Float: {                                                                  \
            using scalar_t_0 = float;                                                                  \
            switch(TYPE1)                                                                              \
            {                                                                                          \
                case at::ScalarType::Float: {                                                          \
                    using scalar_t_1 = float;                                                          \
                    if (REUSE_FREQS_FRONT_PART)                                                        \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = true;                                     \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    else                                                                               \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = false;                                    \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    break;                                                                             \
                }                                                                                      \
                case at::ScalarType::Half: {                                                           \
                    using scalar_t_1 = at::Half;                                                       \
                    if (REUSE_FREQS_FRONT_PART)                                                        \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = true;                                     \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    else                                                                               \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = false;                                    \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    break;                                                                             \
                }                                                                                      \
                case at::ScalarType::BFloat16: {                                                       \
                    using scalar_t_1 = at::BFloat16;                                                   \
                    if (REUSE_FREQS_FRONT_PART)                                                        \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = true;                                     \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    else                                                                               \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = false;                                    \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    break;                                                                             \
                }                                                                                      \
                default:                                                                               \
                    TORCH_CHECK(false, NAME " does't support ",                                        \
                        toString(TYPE0), " with ", toString(TYPE1), ".");                              \
            }                                                                                          \
            break;                                                                                     \
        }                                                                                              \
        case at::ScalarType::Half: {                                                                   \
            using scalar_t_0 = at::Half;                                                               \
            switch(TYPE1)                                                                              \
            {                                                                                          \
                case at::ScalarType::Float: {                                                          \
                    using scalar_t_1 = float;                                                          \
                    if (REUSE_FREQS_FRONT_PART)                                                        \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = true;                                     \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    else                                                                               \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = false;                                    \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    break;                                                                             \
                }                                                                                      \
                case at::ScalarType::Half: {                                                           \
                    using scalar_t_1 = at::Half;                                                       \
                    if (REUSE_FREQS_FRONT_PART)                                                        \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = true;                                     \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    else                                                                               \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = false;                                    \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    break;                                                                             \
                }                                                                                      \
                case at::ScalarType::BFloat16: {                                                       \
                    using scalar_t_1 = at::BFloat16;                                                   \
                    if (REUSE_FREQS_FRONT_PART)                                                        \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = true;                                     \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    else                                                                               \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = false;                                    \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    break;                                                                             \
                }                                                                                      \
                default:                                                                               \
                    TORCH_CHECK(false, NAME " does't support ",                                        \
                        toString(TYPE0), " with ", toString(TYPE1), ".");                              \
            }                                                                                          \
            break;                                                                                     \
        }                                                                                              \
        case at::ScalarType::BFloat16: {                                                               \
            using scalar_t_0 = at::BFloat16;                                                           \
            switch(TYPE1)                                                                              \
            {                                                                                          \
                case at::ScalarType::Float: {                                                          \
                    using scalar_t_1 = float;                                                          \
                    if (REUSE_FREQS_FRONT_PART)                                                        \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = true;                                     \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    else                                                                               \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = false;                                    \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    break;                                                                             \
                }                                                                                      \
                case at::ScalarType::Half: {                                                           \
                    using scalar_t_1 = at::Half;                                                       \
                    if (REUSE_FREQS_FRONT_PART)                                                        \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = true;                                     \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    else                                                                               \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = false;                                    \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    break;                                                                             \
                }                                                                                      \
                case at::ScalarType::BFloat16: {                                                       \
                    using scalar_t_1 = at::BFloat16;                                                   \
                    if (REUSE_FREQS_FRONT_PART)                                                        \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = true;                                     \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    else                                                                               \
                    {                                                                                  \
                        constexpr bool ReuseFreqsFrontPart = false;                                    \
                        if (ROTATE_STYLE == ROTATE_STYLE_NEOX)                                         \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else if (ROTATE_STYLE == ROTATE_STYLE_GPTJ)                                    \
                        {                                                                              \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                         \
                            __VA_ARGS__;                                                               \
                        }                                                                              \
                        else                                                                           \
                        {                                                                              \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                    \
                                        std::to_string(ROTATE_STYLE), ".");                            \
                        }                                                                              \
                    }                                                                                  \
                    break;                                                                             \
                }                                                                                      \
                default:                                                                               \
                    TORCH_CHECK(false, NAME " does't support ",                                        \
                        toString(TYPE0), " with ", toString(TYPE1), ".");                              \
            }                                                                                          \
            break;                                                                                     \
        }                                                                                              \
        default:                                                                                       \
            TORCH_CHECK(false, NAME " does't support ", toString(TYPE0), ".");                         \
    }

// =====================================================================================================================
// Interfaces
//

void rope_fwd_impl(
    torch::Tensor&       output,        // [s, b, h, d]
    const torch::Tensor& input,         // [s, b, h, d]
    const torch::Tensor& freqs,         // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        input.scalar_type(),
        freqs.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_1c_sbhd_uncached<Op1cUncachedFwd, ...>",
        dispatch_1c_sbhd_uncached<Op1cUncachedFwd, RotateStyle, ReuseFreqsFrontPart>(
            output.data_ptr<scalar_t_0>(),
            input.data_ptr<scalar_t_0>(),
            freqs.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_i_s, stride_i_b, stride_i_h, stride_i_d,
            stride_o_s, stride_o_b, stride_o_h, stride_o_d););
}

void rope_bwd_impl(
    torch::Tensor&       input_grads,   // [s, b, h, d]
    const torch::Tensor& output_grads,  // [s, b, h, d]
    const torch::Tensor& freqs,         // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads.scalar_type(),
        freqs.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_1c_sbhd_uncached<Op1cUncachedBwd, ...>",
        dispatch_1c_sbhd_uncached<Op1cUncachedBwd, RotateStyle, ReuseFreqsFrontPart>(
            input_grads.data_ptr<scalar_t_0>(),
            output_grads.data_ptr<scalar_t_0>(),
            freqs.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_o_s, stride_o_b, stride_o_h, stride_o_d,
            stride_i_s, stride_i_b, stride_i_h, stride_i_d););
}

void rope_2c_fwd_impl(
    torch::Tensor&       output_x,      // [s, b, h, d]
    torch::Tensor&       output_y,      // [s, b, h, d]
    const torch::Tensor& input_x,       // [s, b, h, d]
    const torch::Tensor& input_y,       // [s, b, h, d]
    const torch::Tensor& freqs,         // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        input_x.scalar_type(),
        freqs.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_2c_sbhd_uncached<Op2cUncachedFwd, ...>",
        dispatch_2c_sbhd_uncached<Op2cUncachedFwd, RotateStyle, ReuseFreqsFrontPart>(
            output_x.data_ptr<scalar_t_0>(),
            output_y.data_ptr<scalar_t_0>(),
            input_x.data_ptr<scalar_t_0>(),
            input_y.data_ptr<scalar_t_0>(),
            freqs.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d););
}

void rope_2c_bwd_impl(
    torch::Tensor&       input_grads_x, // [s, b, h, d]
    torch::Tensor&       input_grads_y, // [s, b, h, d]
    const torch::Tensor& output_grads_x,// [s, b, h, d]
    const torch::Tensor& output_grads_y,// [s, b, h, d]
    const torch::Tensor& freqs,         // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads_x.scalar_type(),
        freqs.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_2c_sbhd_uncached<Op2cUncachedBwd, ...>",
        dispatch_2c_sbhd_uncached<Op2cUncachedBwd, RotateStyle, ReuseFreqsFrontPart>(
            input_grads_x.data_ptr<scalar_t_0>(),
            input_grads_y.data_ptr<scalar_t_0>(),
            output_grads_x.data_ptr<scalar_t_0>(),
            output_grads_y.data_ptr<scalar_t_0>(),
            freqs.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d););
}

void rope_cached_fwd_impl(
    torch::Tensor&       output,        // [s, b, h, d]
    const torch::Tensor& input,         // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin,           // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        input.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_1c_sbhd_cached<Op1cCachedFwd, ...>",
        dispatch_1c_sbhd_cached<Op1cCachedFwd, RotateStyle, ReuseFreqsFrontPart>(
            output.data_ptr<scalar_t_0>(),
            input.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_i_s, stride_i_b, stride_i_h, stride_i_d,
            stride_o_s, stride_o_b, stride_o_h, stride_o_d););
}

void rope_cached_bwd_impl(
    torch::Tensor&       input_grads,   // [s, b, h, d]
    const torch::Tensor& output_grads,  // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin,           // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_1c_sbhd_cached<Op1cCachedBwd, ...>",
        dispatch_1c_sbhd_cached<Op1cCachedBwd, RotateStyle, ReuseFreqsFrontPart>(
            input_grads.data_ptr<scalar_t_0>(),
            output_grads.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_o_s, stride_o_b, stride_o_h, stride_o_d,
            stride_i_s, stride_i_b, stride_i_h, stride_i_d););
}

void rope_cached_2c_fwd_impl(
    torch::Tensor&       output_x,      // [s, b, h, d]
    torch::Tensor&       output_y,      // [s, b, h, d]
    const torch::Tensor& input_x,       // [s, b, h, d]
    const torch::Tensor& input_y,       // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin,           // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        input_x.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_2c_sbhd_cached<Op2cCachedFwd, ...>",
        dispatch_2c_sbhd_cached<Op2cCachedFwd, RotateStyle, ReuseFreqsFrontPart>(
            output_x.data_ptr<scalar_t_0>(),
            output_y.data_ptr<scalar_t_0>(),
            input_x.data_ptr<scalar_t_0>(),
            input_y.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d););
}

void rope_cached_2c_bwd_impl(
    torch::Tensor&       input_grads_x, // [s, b, h, d]
    torch::Tensor&       input_grads_y, // [s, b, h, d]
    const torch::Tensor& output_grads_x,// [s, b, h, d]
    const torch::Tensor& output_grads_y,// [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin,           // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads_x.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_2c_sbhd_cached<Op2cCachedBwd, ...>",
        dispatch_2c_sbhd_cached<Op2cCachedBwd, RotateStyle, ReuseFreqsFrontPart>(
            input_grads_x.data_ptr<scalar_t_0>(),
            input_grads_y.data_ptr<scalar_t_0>(),
            output_grads_x.data_ptr<scalar_t_0>(),
            output_grads_y.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d););
}

void rope_thd_fwd_impl(
    torch::Tensor&       output,        // [t, h, d]
    const torch::Tensor& input,         // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [b + 1]
    const torch::Tensor& freqs,         // [max_s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        input.scalar_type(),
        freqs.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_1c_thd_uncached<Op1cUncachedFwd, ...>",
        dispatch_1c_thd_uncached<Op1cUncachedFwd, RotateStyle, ReuseFreqsFrontPart>(
            output.data_ptr<scalar_t_0>(),
            input.data_ptr<scalar_t_0>(),
            cu_seqlens.data_ptr<int32_t>(),
            freqs.data_ptr<scalar_t_1>(),
            size_max_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_i_t, stride_i_h, stride_i_d,
            stride_o_t, stride_o_h, stride_o_d););
}

void rope_thd_bwd_impl(
    torch::Tensor&       input_grads,   // [t, h, d]
    const torch::Tensor& output_grads,  // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [b + 1]
    const torch::Tensor& freqs,         // [max_s, 1, 1, d]
    const int            rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads.scalar_type(),
        freqs.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_1c_thd_uncached<Op1cUncachedBwd, ...>",
        dispatch_1c_thd_uncached<Op1cUncachedBwd, RotateStyle, ReuseFreqsFrontPart>(
            input_grads.data_ptr<scalar_t_0>(),
            output_grads.data_ptr<scalar_t_0>(),
            cu_seqlens.data_ptr<int32_t>(),
            freqs.data_ptr<scalar_t_1>(),
            size_max_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_o_t, stride_o_h, stride_o_d,
            stride_i_t, stride_i_h, stride_i_d););
}

void rope_2d_fwd_impl(
    torch::Tensor&       output,
    const torch::Tensor& input,
    const torch::Tensor& cos_h,
    const torch::Tensor& sin_h,
    const torch::Tensor& cos_w,
    const torch::Tensor& sin_w,
    const int32_t        img_height,
    const int32_t        img_width,
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        input.scalar_type(),
        cos_h.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_1c_2d_cached<Op1cCachedFwd, ...>",
        dispatch_1c_2d_cached<Op1cCachedFwd, RotateStyle, ReuseFreqsFrontPart>(
            output.data_ptr<scalar_t_0>(),
            input.data_ptr<scalar_t_0>(),
            cos_h.data_ptr<scalar_t_1>(),
            sin_h.data_ptr<scalar_t_1>(),
            cos_w.data_ptr<scalar_t_1>(),
            sin_w.data_ptr<scalar_t_1>(),
            img_height, img_width,
            size_b, size_h, size_d,
            stride_i_b, stride_i_s, stride_i_h, stride_i_d,
            stride_o_b, stride_o_s, stride_o_h, stride_o_d););
}

void rope_2d_bwd_impl(
    torch::Tensor&       input_grads,
    const torch::Tensor& output_grads,
    const torch::Tensor& cos_h,
    const torch::Tensor& sin_h,
    const torch::Tensor& cos_w,
    const torch::Tensor& sin_w,
    const int32_t        img_height,
    const int32_t        img_width,
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part)
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

    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads.scalar_type(),
        cos_h.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        "dispatch_1c_2d_cached<Op1cCachedBwd, ...>",
        dispatch_1c_2d_cached<Op1cCachedBwd, RotateStyle, ReuseFreqsFrontPart>(
            input_grads.data_ptr<scalar_t_0>(),
            output_grads.data_ptr<scalar_t_0>(),
            cos_h.data_ptr<scalar_t_1>(),
            sin_h.data_ptr<scalar_t_1>(),
            cos_w.data_ptr<scalar_t_1>(),
            sin_w.data_ptr<scalar_t_1>(),
            img_height, img_width,
            size_b, size_h, size_d,
            stride_o_b, stride_o_s, stride_o_h, stride_o_d,
            stride_i_b, stride_i_s, stride_i_h, stride_i_d););
}
