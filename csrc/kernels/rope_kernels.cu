// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <c10/cuda/CUDAGuard.h>
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
// NopeFirst:           [0, size_r(rotate dim)) is rotated and the rest is just copied if this value is false.
//                      [size_d (size of d dim) - size_r, size_d) is rotated and the front part is just copied if true.
//

#define ROTATE_STYLE_NEOX 0
#define ROTATE_STYLE_GPTJ 1

// =====================================================================================================================
// Kernel Helper Functions
//

template <int32_t RotateStyle, bool IsForward, bool ReuseFreqsFrontPart, typename scalar_f_t>
__device__ __forceinline__ void load_cos_sin_uncached(
    float* p_cos_0, float* p_sin_0,
    float* p_cos_1, float* p_sin_1,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t did,
    const int32_t size_half_r)
{
    if constexpr (RotateStyle == ROTATE_STYLE_NEOX)
    {
        if constexpr (IsForward)
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
            const float f_did_0 = float(p_freqs[did]);
            if constexpr (ReuseFreqsFrontPart)
            {
                *p_cos_0 = cosf(f_did_0);
                *p_sin_0 = sinf(f_did_0);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                const float f_did_1 = p_freqs[did + size_half_r];
                *p_cos_0 = cosf(f_did_0);
                *p_sin_0 = sinf(f_did_1);
                *p_cos_1 = cosf(f_did_1);
                *p_sin_1 = sinf(f_did_0);
            }
        }
    }
    else if constexpr (RotateStyle == ROTATE_STYLE_GPTJ)
    {
        if constexpr (IsForward)
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
                sincosf(float(p_freqs[did]), p_sin_0, p_cos_0);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                const float f_did_0 = float(p_freqs[did * 2]);
                const float f_did_1 = float(p_freqs[did * 2 + 1]);
                *p_cos_0 = cosf(f_did_0);
                *p_sin_0 = sinf(f_did_1);
                *p_cos_1 = cosf(f_did_1);
                *p_sin_1 = sinf(f_did_0);
            }
        }
    }
}

template <int32_t RotateStyle, bool IsForward, bool ReuseFreqsFrontPart, typename scalar_f_t>
__device__ __forceinline__ void load_cos_sin_cached(
    float* p_cos_0, float* p_sin_0,
    float* p_cos_1, float* p_sin_1,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t did,
    const int32_t size_half_r)
{
    if constexpr (RotateStyle == ROTATE_STYLE_NEOX)
    {
        if constexpr (IsForward)
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
        if constexpr (IsForward)
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
                *p_cos_0 = float(p_cos[did]);
                *p_sin_0 = float(p_sin[did]);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                *p_cos_0 = float(p_cos[did * 2]);
                *p_sin_0 = float(p_sin[did * 2 + 1]);
                *p_cos_1 = float(p_cos[did * 2 + 1]);
                *p_sin_1 = float(p_sin[did * 2]);
            }
        }
    }
}

template <int32_t RotateStyle, bool StrideDEq1>
__device__ __forceinline__ void get_offset(
    int32_t* p_offset_0,
    int32_t* p_offset_1,
    const int32_t did,
    const int32_t hid,
    const int32_t stride_d,
    const int32_t stride_h,
    const int32_t size_half_r)
{
    const int32_t offset_h = hid * stride_h;

    if constexpr (RotateStyle == ROTATE_STYLE_NEOX)
    {
        *p_offset_0 = offset_h + did * stride_d;
        *p_offset_1 = *p_offset_0 + size_half_r * stride_d;
    }
    else if constexpr (RotateStyle == ROTATE_STYLE_GPTJ)
    {
        *p_offset_0 = offset_h + 2 * did * stride_d;
        if constexpr (StrideDEq1)
        {
            // Asking compiler to merge memory ops when accessing adjacent elements.
            *p_offset_1 = *p_offset_0 + 1;
        }
        else
        {
            *p_offset_1 = *p_offset_0 + stride_d;
        }
    }
}

template <int32_t RotateStyle, bool StrideDEq1, typename o_scalar_t, typename i_scalar_t>
__device__ __forceinline__ void load_payload(
    o_scalar_t*       p_data_0,
    o_scalar_t*       p_data_1,
    const i_scalar_t* p_buffer,
    const int32_t     did,
    const int32_t     hid,
    const int32_t     stride_d,
    const int32_t     stride_h,
    const int32_t     size_half_r)
{
    int32_t offset_0, offset_1;
    get_offset<RotateStyle, StrideDEq1>(&offset_0, &offset_1, did, hid, stride_d, stride_h, size_half_r);

    *p_data_0 = o_scalar_t(p_buffer[offset_0]);
    *p_data_1 = o_scalar_t(p_buffer[offset_1]);
}

template <int32_t RotateStyle, bool StrideDEq1, typename o_scalar_t, typename i_scalar_t>
__device__ __forceinline__ void store_payload(
    o_scalar_t*      p_buffer,
    const i_scalar_t data_0,
    const i_scalar_t data_1,
    const int32_t    did,
    const int32_t    hid,
    const int32_t    stride_d,
    const int32_t    stride_h,
    const int32_t    size_half_r)
{
    int32_t offset_0, offset_1;
    get_offset<RotateStyle, StrideDEq1>(&offset_0, &offset_1, did, hid, stride_d, stride_h, size_half_r);

    p_buffer[offset_0] = o_scalar_t(data_0);
    p_buffer[offset_1] = o_scalar_t(data_1);
}

template <typename scalar_t>
__device__ __forceinline__ void elementwise_copy(
    scalar_t* __restrict__       p_output,
    const scalar_t* __restrict__ p_input,
    const int32_t hid_end,
    const int32_t did_start, const int32_t did_end,
    const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_h, const int32_t stride_o_d)
{
    if (did_end > did_start)
    {
        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < hid_end; hid += blockDim.y)
        {
            const int32_t offset_i = hid * stride_i_h;
            const int32_t offset_o = hid * stride_o_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + did_start; did < did_end; did += blockDim.x)
            {
                p_output[offset_o + did * stride_o_d] = p_input[offset_i + did * stride_i_d];
            }
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void elementwise_copy_2c(
    scalar_t* __restrict__       p_output_x,
    scalar_t* __restrict__       p_output_y,
    const scalar_t* __restrict__ p_input_x,
    const scalar_t* __restrict__ p_input_y,
    const int32_t hid_end_x, const int32_t hid_end_y,
    const int32_t did_start, const int32_t did_end,
    const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    if (did_end > did_start)
    {
        const int32_t hid_min_end = hid_end_x < hid_end_y ? hid_end_x : hid_end_y;

        #pragma unroll
        for (int32_t hid = threadIdx.y; hid < hid_min_end; hid += blockDim.y)
        {
            const int32_t offset_ix = hid * stride_ix_h;
            const int32_t offset_iy = hid * stride_iy_h;
            const int32_t offset_ox = hid * stride_ox_h;
            const int32_t offset_oy = hid * stride_oy_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + did_start; did < did_end; did += blockDim.x)
            {
                p_output_x[offset_ox + did * stride_ox_d] = p_input_x[offset_ix + did * stride_ix_d];
                p_output_y[offset_oy + did * stride_oy_d] = p_input_y[offset_iy + did * stride_iy_d];
            }
        }

        #pragma unroll
        for (int32_t hid = threadIdx.y + hid_min_end; hid < hid_end_x; hid += blockDim.y)
        {
            const int32_t offset_ix = hid * stride_ix_h;
            const int32_t offset_ox = hid * stride_ox_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + did_start; did < did_end; did += blockDim.x)
            {
                p_output_x[offset_ox + did * stride_ox_d] = p_input_x[offset_ix + did * stride_ix_d];
            }
        }

        #pragma unroll
        for (int32_t hid = threadIdx.y + hid_min_end; hid < hid_end_y; hid += blockDim.y)
        {
            const int32_t offset_iy = hid * stride_iy_h;
            const int32_t offset_oy = hid * stride_oy_h;

            #pragma unroll
            for (int32_t did = threadIdx.x + did_start; did < did_end; did += blockDim.x)
            {
                p_output_y[offset_oy + did * stride_oy_d] = p_input_y[offset_iy + did * stride_iy_d];
            }
        }
    }
}

// =====================================================================================================================
// Kernel Functionalities
//

struct OpUncachedFwd
{
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst, bool Inplace,
              bool StrideDOutEq1, bool StrideDInEq1,
              typename scalar_t, typename scalar_f_t>
    __device__ __forceinline__ static void apply_1c(
        scalar_t* __restrict__         p_output,
        const scalar_t* __restrict__   p_input,
        const scalar_f_t* __restrict__ p_freqs,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_i_h, const int32_t stride_i_d,
        const int32_t stride_o_h, const int32_t stride_o_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;

        #pragma unroll
        for (int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_uncached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did - did_start, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                float input_0, input_1;
                load_payload<RotateStyle, StrideDInEq1>(
                    &input_0, &input_1, p_input, did, hid, stride_i_d, stride_i_h, size_half_r);

                const float output_0 = input_0 * cos_0 - input_1 * sin_0;
                const float output_1 = input_1 * cos_1 + input_0 * sin_1;
                store_payload<RotateStyle, StrideDOutEq1>(
                    p_output, output_0, output_1, did, hid, stride_o_d, stride_o_h, size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr (!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy(
                p_output, p_input,
                size_h, did_start, did_end,
                stride_i_h, stride_i_d,
                stride_o_h, stride_o_d);
        }
    }

    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst, bool Inplace,
              bool StrideDOutXEq1, bool StrideDOutYEq1, bool StrideDInXEq1, bool StrideDInYEq1,
              typename scalar_t, typename scalar_f_t>
    __device__ __forceinline__ static void apply_2c(
        scalar_t* __restrict__         p_output_x,
        scalar_t* __restrict__         p_output_y,
        const scalar_t* __restrict__   p_input_x,
        const scalar_t* __restrict__   p_input_y,
        const scalar_f_t* __restrict__ p_freqs,
        const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d, const int32_t size_f,
        const int32_t stride_ix_h, const int32_t stride_ix_d,
        const int32_t stride_iy_h, const int32_t stride_iy_d,
        const int32_t stride_ox_h, const int32_t stride_ox_d,
        const int32_t stride_oy_h, const int32_t stride_oy_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);

        #pragma unroll
        for (int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_uncached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did - did_start, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_min_h; hid += blockDim.y)
            {
                float input_x_0, input_x_1, input_y_0, input_y_1;
                load_payload<RotateStyle, StrideDInXEq1>(
                    &input_x_0, &input_x_1, p_input_x, did, hid, stride_ix_d, stride_ix_h, size_half_r);
                load_payload<RotateStyle, StrideDInYEq1>(
                    &input_y_0, &input_y_1, p_input_y, did, hid, stride_iy_d, stride_iy_h, size_half_r);

                const float output_x_0 = input_x_0 * cos_0 - input_x_1 * sin_0;
                const float output_x_1 = input_x_1 * cos_1 + input_x_0 * sin_1;
                const float output_y_0 = input_y_0 * cos_0 - input_y_1 * sin_0;
                const float output_y_1 = input_y_1 * cos_1 + input_y_0 * sin_1;
                store_payload<RotateStyle, StrideDOutXEq1>(
                    p_output_x, output_x_0, output_x_1, did, hid, stride_ox_d, stride_ox_h, size_half_r);
                store_payload<RotateStyle, StrideDOutYEq1>(
                    p_output_y, output_y_0, output_y_1, did, hid, stride_oy_d, stride_oy_h, size_half_r);
            }

            #pragma unroll
            for (int32_t hid = threadIdx.y + size_min_h; hid < size_h_x; hid += blockDim.y)
            {
                float input_x_0, input_x_1;
                load_payload<RotateStyle, StrideDInXEq1>(
                    &input_x_0, &input_x_1, p_input_x, did, hid, stride_ix_d, stride_ix_h, size_half_r);

                const float output_x_0 = input_x_0 * cos_0 - input_x_1 * sin_0;
                const float output_x_1 = input_x_1 * cos_1 + input_x_0 * sin_1;
                store_payload<RotateStyle, StrideDOutXEq1>(
                    p_output_x, output_x_0, output_x_1, did, hid, stride_ox_d, stride_ox_h, size_half_r);
            }

            #pragma unroll
            for (int32_t hid = threadIdx.y + size_min_h; hid < size_h_y; hid += blockDim.y)
            {
                float input_y_0, input_y_1;
                load_payload<RotateStyle, StrideDInYEq1>(
                    &input_y_0, &input_y_1, p_input_y, did, hid, stride_iy_d, stride_iy_h, size_half_r);

                const float output_y_0 = input_y_0 * cos_0 - input_y_1 * sin_0;
                const float output_y_1 = input_y_1 * cos_1 + input_y_0 * sin_1;
                store_payload<RotateStyle, StrideDOutYEq1>(
                    p_output_y, output_y_0, output_y_1, did, hid, stride_oy_d, stride_oy_h, size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr (!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c(
                p_output_x, p_output_y, p_input_x, p_input_y,
                size_h_x, size_h_y, did_start, did_end,
                stride_ix_h, stride_ix_d, stride_iy_h, stride_iy_d,
                stride_ox_h, stride_ox_d, stride_oy_h, stride_oy_d);
        }
    }
};

struct OpUncachedBwd
{
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst, bool Inplace,
              bool StrideDInGradsEq1, bool StrideDOutGradsEq1,
              typename scalar_t, typename scalar_f_t>
    __device__ __forceinline__ static void apply_1c(
        scalar_t* __restrict__         p_input_grads,
        const scalar_t* __restrict__   p_output_grads,
        const scalar_f_t* __restrict__ p_freqs,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_o_h, const int32_t stride_o_d,
        const int32_t stride_i_h, const int32_t stride_i_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;

        #pragma unroll
        for (int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_uncached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did - did_start, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                float output_grad_0, output_grad_1;
                load_payload<RotateStyle, StrideDOutGradsEq1>(
                    &output_grad_0, &output_grad_1, p_output_grads, did, hid, stride_o_d, stride_o_h, size_half_r);

                const float input_grad_0 = output_grad_0 * cos_0 + output_grad_1 * sin_0;
                const float input_grad_1 = output_grad_1 * cos_1 - output_grad_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsEq1>(
                    p_input_grads, input_grad_0, input_grad_1, did, hid, stride_i_d, stride_i_h, size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr (!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy(
                p_input_grads, p_output_grads,
                size_h, did_start, did_end,
                stride_o_h, stride_o_d,
                stride_i_h, stride_i_d);
        }
    }

    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst, bool Inplace,
              bool StrideDInGradsXEq1, bool StrideDInGradsYEq1, bool StrideDOutGradsXEq1, bool StrideDOutGradsYEq1,
              typename scalar_t, typename scalar_f_t>
    __device__ __forceinline__ static void apply_2c(
        scalar_t* __restrict__         p_input_grads_x,
        scalar_t* __restrict__         p_input_grads_y,
        const scalar_t* __restrict__   p_output_grads_x,
        const scalar_t* __restrict__   p_output_grads_y,
        const scalar_f_t* __restrict__ p_freqs,
        const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d, const int32_t size_f,
        const int32_t stride_ox_h, const int32_t stride_ox_d,
        const int32_t stride_oy_h, const int32_t stride_oy_d,
        const int32_t stride_ix_h, const int32_t stride_ix_d,
        const int32_t stride_iy_h, const int32_t stride_iy_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);

        #pragma unroll
        for (int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_uncached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did - did_start, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_min_h; hid += blockDim.y)
            {
                float output_grad_x_0, output_grad_x_1, output_grad_y_0, output_grad_y_1;
                load_payload<RotateStyle, StrideDOutGradsXEq1>(
                    &output_grad_x_0, &output_grad_x_1, p_output_grads_x, did, hid, stride_ox_d, stride_ox_h, size_half_r);
                load_payload<RotateStyle, StrideDOutGradsYEq1>(
                    &output_grad_y_0, &output_grad_y_1, p_output_grads_y, did, hid, stride_oy_d, stride_oy_h, size_half_r);

                const float input_grad_x_0 = output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0;
                const float input_grad_x_1 = output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1;
                const float input_grad_y_0 = output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0;
                const float input_grad_y_1 = output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsXEq1>(
                    p_input_grads_x, input_grad_x_0, input_grad_x_1, did, hid, stride_ix_d, stride_ix_h, size_half_r);
                store_payload<RotateStyle, StrideDInGradsYEq1>(
                    p_input_grads_y, input_grad_y_0, input_grad_y_1, did, hid, stride_iy_d, stride_iy_h, size_half_r);
            }

            #pragma unroll
            for (int32_t hid = threadIdx.y + size_min_h; hid < size_h_x; hid += blockDim.y)
            {
                float output_grad_x_0, output_grad_x_1;
                load_payload<RotateStyle, StrideDOutGradsXEq1>(
                    &output_grad_x_0, &output_grad_x_1, p_output_grads_x, did, hid, stride_ox_d, stride_ox_h, size_half_r);

                const float input_grad_x_0 = output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0;
                const float input_grad_x_1 = output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsXEq1>(
                    p_input_grads_x, input_grad_x_0, input_grad_x_1, did, hid, stride_ix_d, stride_ix_h, size_half_r);
            }

            #pragma unroll
            for (int32_t hid = threadIdx.y + size_min_h; hid < size_h_y; hid += blockDim.y)
            {
                float output_grad_y_0, output_grad_y_1;
                load_payload<RotateStyle, StrideDOutGradsYEq1>(
                    &output_grad_y_0, &output_grad_y_1, p_output_grads_y, did, hid, stride_oy_d, stride_oy_h, size_half_r);

                const float input_grad_y_0 = output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0;
                const float input_grad_y_1 = output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsYEq1>(
                    p_input_grads_y, input_grad_y_0, input_grad_y_1, did, hid, stride_iy_d, stride_iy_h, size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr (!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c(
                p_input_grads_x, p_input_grads_y, p_output_grads_x, p_output_grads_y,
                size_h_x, size_h_y, did_start, did_end,
                stride_ox_h, stride_ox_d, stride_oy_h, stride_oy_d,
                stride_ix_h, stride_ix_d, stride_iy_h, stride_iy_d);
        }
    }
};

struct OpCachedFwd
{
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst, bool Inplace,
              bool StrideDOutEq1, bool StrideDInEq1,
              typename scalar_t, typename scalar_f_t>
    __device__ __forceinline__ static void apply_1c(
        scalar_t* __restrict__         p_output,
        const scalar_t* __restrict__   p_input,
        const scalar_f_t* __restrict__ p_cos,
        const scalar_f_t* __restrict__ p_sin,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_i_h, const int32_t stride_i_d,
        const int32_t stride_o_h, const int32_t stride_o_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;

        #pragma unroll
        for (int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_cached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did - did_start, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                float input_0, input_1;
                load_payload<RotateStyle, StrideDInEq1>(
                    &input_0, &input_1, p_input, did, hid, stride_i_d, stride_i_h, size_half_r);

                const float output_0 = input_0 * cos_0 - input_1 * sin_0;
                const float output_1 = input_1 * cos_1 + input_0 * sin_1;
                store_payload<RotateStyle, StrideDOutEq1>(
                    p_output, output_0, output_1, did, hid, stride_o_d, stride_o_h, size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr (!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy(
                p_output, p_input,
                size_h, did_start, did_end,
                stride_i_h, stride_i_d,
                stride_o_h, stride_o_d);
        }
    }

    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst, bool Inplace,
              bool StrideDOutXEq1, bool StrideDOutYEq1, bool StrideDInXEq1, bool StrideDInYEq1,
              typename scalar_t, typename scalar_f_t>
    __device__ __forceinline__ static void apply_2c(
        scalar_t* __restrict__         p_output_x,
        scalar_t* __restrict__         p_output_y,
        const scalar_t* __restrict__   p_input_x,
        const scalar_t* __restrict__   p_input_y,
        const scalar_f_t* __restrict__ p_cos,
        const scalar_f_t* __restrict__ p_sin,
        const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d, const int32_t size_f,
        const int32_t stride_ix_h, const int32_t stride_ix_d,
        const int32_t stride_iy_h, const int32_t stride_iy_d,
        const int32_t stride_ox_h, const int32_t stride_ox_d,
        const int32_t stride_oy_h, const int32_t stride_oy_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);

        #pragma unroll
        for (int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_cached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did - did_start, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_min_h; hid += blockDim.y)
            {
                float input_x_0, input_x_1, input_y_0, input_y_1;
                load_payload<RotateStyle, StrideDInXEq1>(
                    &input_x_0, &input_x_1, p_input_x, did, hid, stride_ix_d, stride_ix_h, size_half_r);
                load_payload<RotateStyle, StrideDInYEq1>(
                    &input_y_0, &input_y_1, p_input_y, did, hid, stride_iy_d, stride_iy_h, size_half_r);

                const float output_x_0 = input_x_0 * cos_0 - input_x_1 * sin_0;
                const float output_x_1 = input_x_1 * cos_1 + input_x_0 * sin_1;
                const float output_y_0 = input_y_0 * cos_0 - input_y_1 * sin_0;
                const float output_y_1 = input_y_1 * cos_1 + input_y_0 * sin_1;
                store_payload<RotateStyle, StrideDOutXEq1>(
                    p_output_x, output_x_0, output_x_1, did, hid, stride_ox_d, stride_ox_h, size_half_r);
                store_payload<RotateStyle, StrideDOutYEq1>(
                    p_output_y, output_y_0, output_y_1, did, hid, stride_oy_d, stride_oy_h, size_half_r);
            }

            #pragma unroll
            for (int32_t hid = threadIdx.y + size_min_h; hid < size_h_x; hid += blockDim.y)
            {
                float input_x_0, input_x_1;
                load_payload<RotateStyle, StrideDInXEq1>(
                    &input_x_0, &input_x_1, p_input_x, did, hid, stride_ix_d, stride_ix_h, size_half_r);

                const float output_x_0 = input_x_0 * cos_0 - input_x_1 * sin_0;
                const float output_x_1 = input_x_1 * cos_1 + input_x_0 * sin_1;
                store_payload<RotateStyle, StrideDOutXEq1>(
                    p_output_x, output_x_0, output_x_1, did, hid, stride_ox_d, stride_ox_h, size_half_r);
            }

            #pragma unroll
            for (int32_t hid = threadIdx.y + size_min_h; hid < size_h_y; hid += blockDim.y)
            {
                float input_y_0, input_y_1;
                load_payload<RotateStyle, StrideDInYEq1>(
                    &input_y_0, &input_y_1, p_input_y, did, hid, stride_iy_d, stride_iy_h, size_half_r);

                const float output_y_0 = input_y_0 * cos_0 - input_y_1 * sin_0;
                const float output_y_1 = input_y_1 * cos_1 + input_y_0 * sin_1;
                store_payload<RotateStyle, StrideDOutYEq1>(
                    p_output_y, output_y_0, output_y_1, did, hid, stride_oy_d, stride_oy_h, size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr (!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c(
                p_output_x, p_output_y, p_input_x, p_input_y,
                size_h_x, size_h_y, did_start, did_end,
                stride_ix_h, stride_ix_d, stride_iy_h, stride_iy_d,
                stride_ox_h, stride_ox_d, stride_oy_h, stride_oy_d);
        }
    }
};

struct OpCachedBwd
{
    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst, bool Inplace,
              bool StrideDInGradsEq1, bool StrideDOutGradsEq1,
              typename scalar_t, typename scalar_f_t>
    __device__ __forceinline__ static void apply_1c(
        scalar_t* __restrict__         p_input_grads,
        const scalar_t* __restrict__   p_output_grads,
        const scalar_f_t* __restrict__ p_cos,
        const scalar_f_t* __restrict__ p_sin,
        const int32_t size_h, const int32_t size_d, const int32_t size_f,
        const int32_t stride_o_h, const int32_t stride_o_d,
        const int32_t stride_i_h, const int32_t stride_i_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;

        #pragma unroll
        for (int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_cached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did - did_start, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                float output_grad_0, output_grad_1;
                load_payload<RotateStyle, StrideDOutGradsEq1>(
                    &output_grad_0, &output_grad_1, p_output_grads, did, hid, stride_o_d, stride_o_h, size_half_r);

                const float input_grad_0 = output_grad_0 * cos_0 + output_grad_1 * sin_0;
                const float input_grad_1 = output_grad_1 * cos_1 - output_grad_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsEq1>(
                    p_input_grads, input_grad_0, input_grad_1, did, hid, stride_i_d, stride_i_h, size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr (!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy(
                p_input_grads, p_output_grads,
                size_h, did_start, did_end,
                stride_o_h, stride_o_d,
                stride_i_h, stride_i_d);
        }
    }

    template <int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst, bool Inplace,
              bool StrideDInGradsXEq1, bool StrideDInGradsYEq1, bool StrideDOutGradsXEq1, bool StrideDOutGradsYEq1,
              typename scalar_t, typename scalar_f_t>
    __device__ __forceinline__ static void apply_2c(
        scalar_t* __restrict__         p_input_grads_x,
        scalar_t* __restrict__         p_input_grads_y,
        const scalar_t* __restrict__   p_output_grads_x,
        const scalar_t* __restrict__   p_output_grads_y,
        const scalar_f_t* __restrict__ p_cos,
        const scalar_f_t* __restrict__ p_sin,
        const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d, const int32_t size_f,
        const int32_t stride_ox_h, const int32_t stride_ox_d,
        const int32_t stride_oy_h, const int32_t stride_oy_d,
        const int32_t stride_ix_h, const int32_t stride_ix_d,
        const int32_t stride_iy_h, const int32_t stride_iy_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);

        #pragma unroll
        for (int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_cached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did - did_start, size_half_r);

            #pragma unroll
            for (int32_t hid = threadIdx.y; hid < size_min_h; hid += blockDim.y)
            {
                float output_grad_x_0, output_grad_x_1, output_grad_y_0, output_grad_y_1;
                load_payload<RotateStyle, StrideDOutGradsXEq1>(
                    &output_grad_x_0, &output_grad_x_1, p_output_grads_x, did, hid, stride_ox_d, stride_ox_h, size_half_r);
                load_payload<RotateStyle, StrideDOutGradsYEq1>(
                    &output_grad_y_0, &output_grad_y_1, p_output_grads_y, did, hid, stride_oy_d, stride_oy_h, size_half_r);

                const float input_grad_x_0 = output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0;
                const float input_grad_x_1 = output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1;
                const float input_grad_y_0 = output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0;
                const float input_grad_y_1 = output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsXEq1>(
                    p_input_grads_x, input_grad_x_0, input_grad_x_1, did, hid, stride_ix_d, stride_ix_h, size_half_r);
                store_payload<RotateStyle, StrideDInGradsYEq1>(
                    p_input_grads_y, input_grad_y_0, input_grad_y_1, did, hid, stride_iy_d, stride_iy_h, size_half_r);
            }

            #pragma unroll
            for (int32_t hid = threadIdx.y + size_min_h; hid < size_h_x; hid += blockDim.y)
            {
                float output_grad_x_0, output_grad_x_1;
                load_payload<RotateStyle, StrideDOutGradsXEq1>(
                    &output_grad_x_0, &output_grad_x_1, p_output_grads_x, did, hid, stride_ox_d, stride_ox_h, size_half_r);
                
                const float input_grad_x_0 = output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0;
                const float input_grad_x_1 = output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsXEq1>(
                    p_input_grads_x, input_grad_x_0, input_grad_x_1, did, hid, stride_ix_d, stride_ix_h, size_half_r);
            }

            #pragma unroll
            for (int32_t hid = threadIdx.y + size_min_h; hid < size_h_y; hid += blockDim.y)
            {
                float output_grad_y_0, output_grad_y_1;
                load_payload<RotateStyle, StrideDOutGradsYEq1>(
                    &output_grad_y_0, &output_grad_y_1, p_output_grads_y, did, hid, stride_oy_d, stride_oy_h, size_half_r);

                const float input_grad_y_0 = output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0;
                const float input_grad_y_1 = output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsYEq1>(
                    p_input_grads_y, input_grad_y_0, input_grad_y_1, did, hid, stride_iy_d, stride_iy_h, size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr (!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c(
                p_input_grads_x, p_input_grads_y, p_output_grads_x, p_output_grads_y,
                size_h_x, size_h_y, did_start, did_end,
                stride_ox_h, stride_ox_d, stride_oy_h, stride_oy_d,
                stride_ix_h, stride_ix_d, stride_iy_h, stride_iy_d);
        }
    }
};

// =====================================================================================================================
// Kernel Entries
//

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDOutEq1, bool StrideDInEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_1c_sbhd_uncached(
    scalar_t* __restrict__         p_output,
    const scalar_t* __restrict__   p_input,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_i_s, const int32_t stride_i_b, const int32_t stride_i_h, const int32_t stride_i_d,
    const int32_t stride_o_s, const int32_t stride_o_b, const int32_t stride_o_h, const int32_t stride_o_d)
{
    const int32_t sid      = blockIdx.x;
    const int32_t bid      = blockIdx.y;
    const int32_t offset_i = sid * stride_i_s + bid * stride_i_b;
    const int32_t offset_o = sid * stride_o_s + bid * stride_o_b;
    const int32_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, false, StrideDOutEq1, StrideDInEq1>(
        p_output + offset_o,
        p_input + offset_i,
        p_freqs + offset_f,
        size_h, size_d, size_f,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_1c_sbhd_uncached_inplace(
    scalar_t* __restrict__         p_inout,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_s, const int32_t stride_b, const int32_t stride_h, const int32_t stride_d)
{
    const int32_t sid      = blockIdx.x;
    const int32_t bid      = blockIdx.y;
    const int32_t offset   = sid * stride_s + bid * stride_b;
    const int32_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, true, StrideDEq1, StrideDEq1>(
        p_inout + offset,
        p_inout + offset,
        p_freqs + offset_f,
        size_h, size_d, size_f,
        stride_h, stride_d,
        stride_h, stride_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDOutXEq1, bool StrideDOutYEq1, bool StrideDInXEq1, bool StrideDInYEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_uncached(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const int32_t sid       = blockIdx.x;
    const int32_t bid       = blockIdx.y;
    const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const int32_t offset_f  = sid * size_f;

    Op::template apply_2c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, false, StrideDOutXEq1, StrideDOutYEq1, StrideDInXEq1, StrideDInYEq1>(
        p_output_x + offset_ox,
        p_output_y + offset_oy,
        p_input_x + offset_ix,
        p_input_y + offset_iy,
        p_freqs + offset_f,
        size_h_x, size_h_y, size_d, size_f,
        stride_ix_h, stride_ix_d,
        stride_iy_h, stride_iy_d,
        stride_ox_h, stride_ox_d,
        stride_oy_h, stride_oy_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDXEq1, bool StrideDYEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_uncached_inplace(
    scalar_t* __restrict__         p_inout_x,
    scalar_t* __restrict__         p_inout_y,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_x_s, const int32_t stride_x_b, const int32_t stride_x_h, const int32_t stride_x_d,
    const int32_t stride_y_s, const int32_t stride_y_b, const int32_t stride_y_h, const int32_t stride_y_d)
{
    const int32_t sid      = blockIdx.x;
    const int32_t bid      = blockIdx.y;
    const int32_t offset_x = sid * stride_x_s + bid * stride_x_b;
    const int32_t offset_y = sid * stride_y_s + bid * stride_y_b;
    const int32_t offset_f = sid * size_f;

    Op::template apply_2c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, true, StrideDXEq1, StrideDYEq1, StrideDXEq1, StrideDYEq1>(
        p_inout_x + offset_x,
        p_inout_y + offset_y,
        p_inout_x + offset_x,
        p_inout_y + offset_y,
        p_freqs + offset_f,
        size_h_x, size_h_y, size_d, size_f,
        stride_x_h, stride_x_d,
        stride_y_h, stride_y_d,
        stride_x_h, stride_x_d,
        stride_y_h, stride_y_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDOutEq1, bool StrideDInEq1,
          typename scalar_t, typename scalar_f_t>
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
    const int32_t sid      = blockIdx.x;
    const int32_t bid      = blockIdx.y;
    const int32_t offset_i = sid * stride_i_s + bid * stride_i_b;
    const int32_t offset_o = sid * stride_o_s + bid * stride_o_b;
    const int32_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, false, StrideDOutEq1, StrideDInEq1>(
        p_output + offset_o,
        p_input + offset_i,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h, size_d, size_f,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_1c_sbhd_cached_inplace(
    scalar_t* __restrict__         p_inout,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_s, const int32_t stride_b, const int32_t stride_h, const int32_t stride_d)
{
    const int32_t sid      = blockIdx.x;
    const int32_t bid      = blockIdx.y;
    const int32_t offset   = sid * stride_s + bid * stride_b;
    const int32_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, true, StrideDEq1, StrideDEq1>(
        p_inout + offset,
        p_inout + offset,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h, size_d, size_f,
        stride_h, stride_d,
        stride_h, stride_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDOutXEq1, bool StrideDOutYEq1, bool StrideDInXEq1, bool StrideDInYEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_cached(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const int32_t sid       = blockIdx.x;
    const int32_t bid       = blockIdx.y;
    const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const int32_t offset_f  = sid * size_f;

    Op::template apply_2c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, false, StrideDOutXEq1, StrideDOutYEq1, StrideDInXEq1, StrideDInYEq1>(
        p_output_x + offset_ox,
        p_output_y + offset_oy,
        p_input_x + offset_ix,
        p_input_y + offset_iy,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h_x, size_h_y, size_d, size_f,
        stride_ix_h, stride_ix_d,
        stride_iy_h, stride_iy_d,
        stride_ox_h, stride_ox_d,
        stride_oy_h, stride_oy_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDXEq1, bool StrideDYEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_cached_inplace(
    scalar_t* __restrict__         p_inout_x,
    scalar_t* __restrict__         p_inout_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_x_s, const int32_t stride_x_b, const int32_t stride_x_h, const int32_t stride_x_d,
    const int32_t stride_y_s, const int32_t stride_y_b, const int32_t stride_y_h, const int32_t stride_y_d)
{
    const int32_t sid      = blockIdx.x;
    const int32_t bid      = blockIdx.y;
    const int32_t offset_x = sid * stride_x_s + bid * stride_x_b;
    const int32_t offset_y = sid * stride_y_s + bid * stride_y_b;
    const int32_t offset_f = sid * size_f;

    Op::template apply_2c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, true, StrideDXEq1, StrideDYEq1, StrideDXEq1, StrideDYEq1>(
        p_inout_x + offset_x,
        p_inout_y + offset_y,
        p_inout_x + offset_x,
        p_inout_y + offset_y,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h_x, size_h_y, size_d, size_f,
        stride_x_h, stride_x_d,
        stride_y_h, stride_y_d,
        stride_x_h, stride_x_d,
        stride_y_h, stride_y_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDOutXEq1, bool StrideDOutYEq1, bool StrideDInXEq1, bool StrideDInYEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_cached_indirect(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__    p_indirect_buffer,
    const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,       // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const int32_t sid       = blockIdx.x;
    const int32_t bid       = blockIdx.y;
    const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const int32_t ib_idx    = sid * gridDim.y + bid;
    const int64_t offset_f  = p_indirect_buffer[ib_idx] * size_f;

    Op::template apply_2c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, false, StrideDOutXEq1, StrideDOutYEq1, StrideDInXEq1, StrideDInYEq1>(
        p_output_x + offset_ox,
        p_output_y + offset_oy,
        p_input_x + offset_ix,
        p_input_y + offset_iy,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h_x, size_h_y, size_d, size_f,
        stride_ix_h, stride_ix_d,
        stride_iy_h, stride_iy_d,
        stride_ox_h, stride_ox_d,
        stride_oy_h, stride_oy_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDXEq1, bool StrideDYEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_cached_indirect_inplace(
    scalar_t* __restrict__         p_inout_x,
    scalar_t* __restrict__         p_inout_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__    p_indirect_buffer,
    const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,       // size of last dimension of freqs.
    const int32_t stride_x_s, const int32_t stride_x_b, const int32_t stride_x_h, const int32_t stride_x_d,
    const int32_t stride_y_s, const int32_t stride_y_b, const int32_t stride_y_h, const int32_t stride_y_d)
{
    const int32_t sid      = blockIdx.x;
    const int32_t bid      = blockIdx.y;
    const int32_t offset_x = sid * stride_x_s + bid * stride_x_b;
    const int32_t offset_y = sid * stride_y_s + bid * stride_y_b;
    const int32_t ib_idx    = sid * gridDim.y + bid;
    const int64_t offset_f = p_indirect_buffer[ib_idx] * size_f;

    Op::template apply_2c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, true, StrideDXEq1, StrideDYEq1, StrideDXEq1, StrideDYEq1>(
        p_inout_x + offset_x,
        p_inout_y + offset_y,
        p_inout_x + offset_x,
        p_inout_y + offset_y,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h_x, size_h_y, size_d, size_f,
        stride_x_h, stride_x_d,
        stride_y_h, stride_y_d,
        stride_x_h, stride_x_d,
        stride_y_h, stride_y_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDOutXEq1, bool StrideDOutYEq1, bool StrideDInXEq1, bool StrideDInYEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_cached_indirect2(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__    p_indirect_buffer_0,
    const int64_t* __restrict__    p_indirect_buffer_1,
    const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,       // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const int32_t sid       = blockIdx.x;
    const int32_t bid       = blockIdx.y;
    const int32_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const int32_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const int32_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const int32_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const int32_t ib_idx    = sid * gridDim.y + bid;
    const int64_t offset_f  = (p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx]) * size_f;

    Op::template apply_2c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, false, StrideDOutXEq1, StrideDOutYEq1, StrideDInXEq1, StrideDInYEq1>(
        p_output_x + offset_ox,
        p_output_y + offset_oy,
        p_input_x + offset_ix,
        p_input_y + offset_iy,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h_x, size_h_y, size_d, size_f,
        stride_ix_h, stride_ix_d,
        stride_iy_h, stride_iy_d,
        stride_ox_h, stride_ox_d,
        stride_oy_h, stride_oy_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDXEq1, bool StrideDYEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_2c_sbhd_cached_indirect2_inplace(
    scalar_t* __restrict__         p_inout_x,
    scalar_t* __restrict__         p_inout_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__    p_indirect_buffer_0,
    const int64_t* __restrict__    p_indirect_buffer_1,
    const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,       // size of last dimension of freqs.
    const int32_t stride_x_s, const int32_t stride_x_b, const int32_t stride_x_h, const int32_t stride_x_d,
    const int32_t stride_y_s, const int32_t stride_y_b, const int32_t stride_y_h, const int32_t stride_y_d)
{
    const int32_t sid       = blockIdx.x;
    const int32_t bid       = blockIdx.y;
    const int32_t offset_x  = sid * stride_x_s + bid * stride_x_b;
    const int32_t offset_y  = sid * stride_y_s + bid * stride_y_b;
    const int32_t ib_idx    = sid * gridDim.y + bid;
    const int64_t offset_f  = (p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx]) * size_f;

    Op::template apply_2c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, true, StrideDXEq1, StrideDYEq1, StrideDXEq1, StrideDYEq1>(
        p_inout_x + offset_x,
        p_inout_y + offset_y,
        p_inout_x + offset_x,
        p_inout_y + offset_y,
        p_cos + offset_f,
        p_sin + offset_f,
        size_h_x, size_h_y, size_d, size_f,
        stride_x_h, stride_x_d,
        stride_y_h, stride_y_d,
        stride_x_h, stride_x_d,
        stride_y_h, stride_y_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDOutEq1, bool StrideDInEq1,
          typename scalar_t, typename scalar_f_t>
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

        Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, false, StrideDOutEq1, StrideDInEq1>(
            p_output + offset_o,
            p_input + offset_i,
            p_freqs + offset_f,
            size_h, size_d, size_f,
            stride_i_h, stride_i_d,
            stride_o_h, stride_o_d);
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_1c_thd_uncached_inplace(
    scalar_t* __restrict__         p_inout,
    const int32_t* __restrict__    p_cu_seqlens,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_h, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_t, const int32_t stride_h, const int32_t stride_d)
{
    const int32_t sid = blockIdx.x;
    const int32_t bid = blockIdx.y;
    const int32_t tid = sid + p_cu_seqlens[bid];

    if (tid < p_cu_seqlens[bid + 1])
    {
        const int32_t offset   = tid * stride_t;
        const int32_t offset_f = sid * size_f;

        Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, true, StrideDEq1, StrideDEq1>(
            p_inout + offset,
            p_inout + offset,
            p_freqs + offset_f,
            size_h, size_d, size_f,
            stride_h, stride_d,
            stride_h, stride_d);
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDOutEq1, bool StrideDInEq1,
          typename scalar_t, typename scalar_f_t>
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
    Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, false, StrideDOutEq1, StrideDInEq1>(
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
    Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, false, StrideDOutEq1, StrideDInEq1>(
        p_output + offset_w_o,
        p_input + offset_w_i,
        p_cos_w + offset_w_f,
        p_sin_w + offset_w_f,
        size_h, size_half_d, size_half_d,
        stride_i_h, stride_i_d,
        stride_o_h, stride_o_d);
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t, typename scalar_f_t>
__global__ void kn_entry_1c_2d_cached_inplace(
    scalar_t* __restrict__         p_inout,
    const scalar_f_t* __restrict__ p_cos_h,
    const scalar_f_t* __restrict__ p_sin_h,
    const scalar_f_t* __restrict__ p_cos_w,
    const scalar_f_t* __restrict__ p_sin_w,
    const int32_t img_width, const int32_t size_h, const int32_t size_d,
    const int32_t stride_b, const int32_t stride_s, const int32_t stride_h, const int32_t stride_d)
{
    const int Hid = blockIdx.x;
    const int Wid = blockIdx.y;
    const int sid = Hid * img_width + Wid;
    const int bid = blockIdx.z;
    const int size_half_d = size_d >> 1;

    const int offset_h   = bid * stride_b + sid * stride_s;
    const int offset_h_f = Hid * size_half_d;
    Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, true, StrideDEq1, StrideDEq1>(
        p_inout + offset_h,
        p_inout + offset_h,
        p_cos_h + offset_h_f,
        p_sin_h + offset_h_f,
        size_h, size_half_d, size_half_d,
        stride_h, stride_d,
        stride_h, stride_d);

    const int offset_w   = offset_h + size_half_d * stride_d;
    const int offset_w_f = Wid * size_half_d;
    Op::template apply_1c<RotateStyle, ReuseFreqsFrontPart, NopeFirst, true, StrideDEq1, StrideDEq1>(
        p_inout + offset_w,
        p_inout + offset_w,
        p_cos_w + offset_w_f,
        p_sin_w + offset_w_f,
        size_h, size_half_d, size_half_d,
        stride_h, stride_d,
        stride_h, stride_d);
}

// =====================================================================================================================
// Dispatches
//

#define LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(ROTATE_STYLE, STRIDE_0, ...)                                 \
    if constexpr ((ROTATE_STYLE) != ROTATE_STYLE_GPTJ)                                                      \
    {                                                                                                       \
        constexpr bool Stride0Eq1 = false;                                                                  \
        __VA_ARGS__;                                                                                        \
    }                                                                                                       \
    else if ((STRIDE_0) == 1)                                                                               \
    {                                                                                                       \
        constexpr bool Stride0Eq1 = true;                                                                   \
        __VA_ARGS__;                                                                                        \
    }                                                                                                       \
    else                                                                                                    \
    {                                                                                                       \
        constexpr bool Stride0Eq1 = false;                                                                  \
        __VA_ARGS__;                                                                                        \
    }

#define LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(ROTATE_STYLE, STRIDE_0, STRIDE_1, ...)                       \
    if constexpr ((ROTATE_STYLE) != ROTATE_STYLE_GPTJ)                                                      \
    {                                                                                                       \
        constexpr bool Stride0Eq1 = false;                                                                  \
        constexpr bool Stride1Eq1 = false;                                                                  \
        __VA_ARGS__;                                                                                        \
    }                                                                                                       \
    else if ((STRIDE_0) == 1)                                                                               \
    {                                                                                                       \
        constexpr bool Stride0Eq1 = true;                                                                   \
        if ((STRIDE_1) == 1)                                                                                \
        {                                                                                                   \
            constexpr bool Stride1Eq1 = true;                                                               \
            __VA_ARGS__;                                                                                    \
        }                                                                                                   \
        else                                                                                                \
        {                                                                                                   \
            constexpr bool Stride1Eq1 = false;                                                              \
            __VA_ARGS__;                                                                                    \
        }                                                                                                   \
    }                                                                                                       \
    else                                                                                                    \
    {                                                                                                       \
        constexpr bool Stride0Eq1 = false;                                                                  \
        if ((STRIDE_1) == 1)                                                                                \
        {                                                                                                   \
            constexpr bool Stride1Eq1 = true;                                                               \
            __VA_ARGS__;                                                                                    \
        }                                                                                                   \
        else                                                                                                \
        {                                                                                                   \
            constexpr bool Stride1Eq1 = false;                                                              \
            __VA_ARGS__;                                                                                    \
        }                                                                                                   \
    }

#define LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(ROTATE_STYLE, STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3, ...)   \
    if constexpr ((ROTATE_STYLE) != ROTATE_STYLE_GPTJ)                                                      \
    {                                                                                                       \
        constexpr bool Stride0Eq1 = false;                                                                  \
        constexpr bool Stride1Eq1 = false;                                                                  \
        constexpr bool Stride2Eq1 = false;                                                                  \
        constexpr bool Stride3Eq1 = false;                                                                  \
        __VA_ARGS__;                                                                                        \
    }                                                                                                       \
    else if ((STRIDE_0) == 1)                                                                               \
    {                                                                                                       \
        constexpr bool Stride0Eq1 = true;                                                                   \
        if ((STRIDE_1) == 1)                                                                                \
        {                                                                                                   \
            constexpr bool Stride1Eq1 = true;                                                               \
            if ((STRIDE_2) == 1)                                                                            \
            {                                                                                               \
                constexpr bool Stride2Eq1 = true;                                                           \
                if ((STRIDE_3) == 1)                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = true;                                                       \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
                else                                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = false;                                                      \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
            }                                                                                               \
            else                                                                                            \
            {                                                                                               \
                constexpr bool Stride2Eq1 = false;                                                          \
                if ((STRIDE_3) == 1)                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = true;                                                       \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
                else                                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = false;                                                      \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        else                                                                                                \
        {                                                                                                   \
            constexpr bool Stride1Eq1 = false;                                                              \
            if ((STRIDE_2) == 1)                                                                            \
            {                                                                                               \
                constexpr bool Stride2Eq1 = true;                                                           \
                if ((STRIDE_3) == 1)                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = true;                                                       \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
                else                                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = false;                                                      \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
            }                                                                                               \
            else                                                                                            \
            {                                                                                               \
                constexpr bool Stride2Eq1 = false;                                                          \
                if ((STRIDE_3) == 1)                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = true;                                                       \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
                else                                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = false;                                                      \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
    }                                                                                                       \
    else                                                                                                    \
    {                                                                                                       \
        constexpr bool Stride0Eq1 = false;                                                                  \
        if ((STRIDE_1) == 1)                                                                                \
        {                                                                                                   \
            constexpr bool Stride1Eq1 = true;                                                               \
            if ((STRIDE_2) == 1)                                                                            \
            {                                                                                               \
                constexpr bool Stride2Eq1 = true;                                                           \
                if ((STRIDE_3) == 1)                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = true;                                                       \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
                else                                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = false;                                                      \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
            }                                                                                               \
            else                                                                                            \
            {                                                                                               \
                constexpr bool Stride2Eq1 = false;                                                          \
                if ((STRIDE_3) == 1)                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = true;                                                       \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
                else                                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = false;                                                      \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        else                                                                                                \
        {                                                                                                   \
            constexpr bool Stride1Eq1 = false;                                                              \
            if ((STRIDE_2) == 1)                                                                            \
            {                                                                                               \
                constexpr bool Stride2Eq1 = true;                                                           \
                if ((STRIDE_3) == 1)                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = true;                                                       \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
                else                                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = false;                                                      \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
            }                                                                                               \
            else                                                                                            \
            {                                                                                               \
                constexpr bool Stride2Eq1 = false;                                                          \
                if ((STRIDE_3) == 1)                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = true;                                                       \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
                else                                                                                        \
                {                                                                                           \
                    constexpr bool Stride3Eq1 = false;                                                      \
                    __VA_ARGS__;                                                                            \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
    }

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          typename scalar_t, typename scalar_f_t>
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

    if (p_output == p_input)
    {
        assert(stride_i_s == stride_o_s);
        assert(stride_i_b == stride_o_b);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
            RotateStyle,
            stride_i_d,
            kn_entry_1c_sbhd_uncached_inplace<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1><<<grid, block, 0, stream>>>(
                p_output,
                p_freqs,
                size_h, size_d, size_f,
                stride_i_s, stride_i_b, stride_i_h, stride_i_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
            RotateStyle,
            stride_i_d,
            stride_o_d,
            kn_entry_1c_sbhd_uncached<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1><<<grid, block, 0, stream>>>(
                p_output,
                p_input,
                p_freqs,
                size_h, size_d, size_f,
                stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                stride_o_s, stride_o_b, stride_o_h, stride_o_d);
        );
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          typename scalar_t, typename scalar_f_t>
void dispatch_2c_sbhd_uncached(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_freqs,
    const int32_t size_s, const int32_t size_b, const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h_x < 16 ? 4 : 8);

    if ((p_output_x == p_input_x) && (p_output_y == p_input_y))
    {
        assert(stride_ix_s == stride_ox_s);
        assert(stride_ix_b == stride_ox_b);
        assert(stride_ix_h == stride_ox_h);
        assert(stride_ix_d == stride_ox_d);
        assert(stride_iy_s == stride_oy_s);
        assert(stride_iy_b == stride_oy_b);
        assert(stride_iy_h == stride_oy_h);
        assert(stride_iy_d == stride_oy_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
            RotateStyle,
            stride_ix_d,
            stride_iy_d,
            kn_entry_2c_sbhd_uncached_inplace<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1><<<grid, block, 0, stream>>>(
                p_output_x,
                p_output_y,
                p_freqs,
                size_h_x, size_h_y, size_d, size_f,
                stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
        );
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(
            RotateStyle,
            stride_ix_d,
            stride_iy_d,
            stride_ox_d,
            stride_oy_d,
            kn_entry_2c_sbhd_uncached<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1, Stride2Eq1, Stride3Eq1><<<grid, block, 0, stream>>>(
                p_output_x,
                p_output_y,
                p_input_x,
                p_input_y,
                p_freqs,
                size_h_x, size_h_y, size_d, size_f,
                stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
        );
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          typename scalar_t, typename scalar_f_t>
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

    if (p_output == p_input)
    {
        assert(stride_i_s == stride_o_s);
        assert(stride_i_b == stride_o_b);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
            RotateStyle,
            stride_i_d,
            kn_entry_1c_sbhd_cached_inplace<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1><<<grid, block, 0, stream>>>(
                p_output,
                p_cos, p_sin,
                size_h, size_d, size_f,
                stride_i_s, stride_i_b, stride_i_h, stride_i_d);
        );
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
            RotateStyle,
            stride_i_d,
            stride_o_d,
            kn_entry_1c_sbhd_cached<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1><<<grid, block, 0, stream>>>(
                p_output,
                p_input,
                p_cos, p_sin,
                size_h, size_d, size_f,
                stride_i_s, stride_i_b, stride_i_h, stride_i_d,
                stride_o_s, stride_o_b, stride_o_h, stride_o_d);
        );
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          typename scalar_t, typename scalar_f_t>
void dispatch_2c_sbhd_cached(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int32_t size_s, const int32_t size_b, const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,   // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h_x < 16 ? 4 : 8);

    if ((p_output_x == p_input_x) && (p_output_y == p_input_y))
    {
        assert(stride_ix_s == stride_ox_s);
        assert(stride_ix_b == stride_ox_b);
        assert(stride_ix_h == stride_ox_h);
        assert(stride_ix_d == stride_ox_d);
        assert(stride_iy_s == stride_oy_s);
        assert(stride_iy_b == stride_oy_b);
        assert(stride_iy_h == stride_oy_h);
        assert(stride_iy_d == stride_oy_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
            RotateStyle,
            stride_ix_d,
            stride_iy_d,
            kn_entry_2c_sbhd_cached_inplace<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1><<<grid, block, 0, stream>>>(
                p_output_x,
                p_output_y,
                p_cos, p_sin,
                size_h_x, size_h_y, size_d, size_f,
                stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
        );
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(
            RotateStyle,
            stride_ix_d,
            stride_iy_d,
            stride_ox_d,
            stride_oy_d,
            kn_entry_2c_sbhd_cached<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1, Stride2Eq1, Stride3Eq1><<<grid, block, 0, stream>>>(
                p_output_x,
                p_output_y,
                p_input_x,
                p_input_y,
                p_cos, p_sin,
                size_h_x, size_h_y, size_d, size_f,
                stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
        );
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          typename scalar_t, typename scalar_f_t>
void dispatch_2c_sbhd_cached_indirect(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__    p_indirect_buffer,
    const int32_t size_s, const int32_t size_b, const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,       // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h_x < 16 ? 4 : 8);

    if ((p_output_x == p_input_x) && (p_output_y == p_input_y))
    {
        assert(stride_ix_s == stride_ox_s);
        assert(stride_ix_b == stride_ox_b);
        assert(stride_ix_h == stride_ox_h);
        assert(stride_ix_d == stride_ox_d);
        assert(stride_iy_s == stride_oy_s);
        assert(stride_iy_b == stride_oy_b);
        assert(stride_iy_h == stride_oy_h);
        assert(stride_iy_d == stride_oy_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
            RotateStyle,
            stride_ix_d,
            stride_iy_d,
            kn_entry_2c_sbhd_cached_indirect_inplace<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1><<<grid, block, 0, stream>>>(
                p_output_x,
                p_output_y,
                p_cos, p_sin,
                p_indirect_buffer,
                size_h_x, size_h_y, size_d, size_f,
                stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
        );
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(
            RotateStyle,
            stride_ix_d,
            stride_iy_d,
            stride_ox_d,
            stride_oy_d,
            kn_entry_2c_sbhd_cached_indirect<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1, Stride2Eq1, Stride3Eq1><<<grid, block, 0, stream>>>(
                p_output_x,
                p_output_y,
                p_input_x,
                p_input_y,
                p_cos, p_sin,
                p_indirect_buffer,
                size_h_x, size_h_y, size_d, size_f,
                stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
        );
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          typename scalar_t, typename scalar_f_t>
void dispatch_2c_sbhd_cached_indirect2(
    scalar_t* __restrict__         p_output_x,
    scalar_t* __restrict__         p_output_y,
    const scalar_t* __restrict__   p_input_x,
    const scalar_t* __restrict__   p_input_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__    p_indirect_buffer_0,
    const int64_t* __restrict__    p_indirect_buffer_1,
    const int32_t size_s, const int32_t size_b, const int32_t size_h_x, const int32_t size_h_y, const int32_t size_d,
    const int32_t size_f,       // size of last dimension of freqs.
    const int32_t stride_ix_s, const int32_t stride_ix_b, const int32_t stride_ix_h, const int32_t stride_ix_d,
    const int32_t stride_iy_s, const int32_t stride_iy_b, const int32_t stride_iy_h, const int32_t stride_iy_d,
    const int32_t stride_ox_s, const int32_t stride_ox_b, const int32_t stride_ox_h, const int32_t stride_ox_d,
    const int32_t stride_oy_s, const int32_t stride_oy_b, const int32_t stride_oy_h, const int32_t stride_oy_d)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(size_s, size_b);
    const dim3 block(C10_WARP_SIZE, size_h_x < 16 ? 4 : 8);

    if ((p_output_x == p_input_x) && (p_output_y == p_input_y))
    {
        assert(stride_ix_s == stride_ox_s);
        assert(stride_ix_b == stride_ox_b);
        assert(stride_ix_h == stride_ox_h);
        assert(stride_ix_d == stride_ox_d);
        assert(stride_iy_s == stride_oy_s);
        assert(stride_iy_b == stride_oy_b);
        assert(stride_iy_h == stride_oy_h);
        assert(stride_iy_d == stride_oy_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
            RotateStyle,
            stride_ix_d,
            stride_iy_d,
            kn_entry_2c_sbhd_cached_indirect2_inplace<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1><<<grid, block, 0, stream>>>(
                p_output_x,
                p_output_y,
                p_cos, p_sin,
                p_indirect_buffer_0,
                p_indirect_buffer_1,
                size_h_x, size_h_y, size_d, size_f,
                stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d);
        );
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(
            RotateStyle,
            stride_ix_d,
            stride_iy_d,
            stride_ox_d,
            stride_oy_d,
            kn_entry_2c_sbhd_cached_indirect2<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1, Stride2Eq1, Stride3Eq1><<<grid, block, 0, stream>>>(
                p_output_x,
                p_output_y,
                p_input_x,
                p_input_y,
                p_cos, p_sin,
                p_indirect_buffer_0,
                p_indirect_buffer_1,
                size_h_x, size_h_y, size_d, size_f,
                stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
                stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
                stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
                stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d);
        );
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          typename scalar_t, typename scalar_f_t>
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

    if (p_output == p_input)
    {
        assert(stride_i_t == stride_o_t);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
            RotateStyle,
            stride_i_d,
            kn_entry_1c_thd_uncached_inplace<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1><<<grid, block, 0, stream>>>(
                p_output,
                p_cu_seqlens,
                p_freqs,
                size_h, size_d, size_f,
                stride_i_t, stride_i_h, stride_i_d);
        );
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
            RotateStyle,
            stride_i_d,
            stride_o_d,
            kn_entry_1c_thd_uncached<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1><<<grid, block, 0, stream>>>(
                p_output,
                p_input,
                p_cu_seqlens,
                p_freqs,
                size_h, size_d, size_f,
                stride_i_t, stride_i_h, stride_i_d,
                stride_o_t, stride_o_h, stride_o_d);
        );
    }
}

template <typename Op, int32_t RotateStyle, bool ReuseFreqsFrontPart, bool NopeFirst,
          typename scalar_t, typename scalar_f_t>
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

    if (p_output == p_input)
    {
        assert(stride_i_s == stride_o_s);
        assert(stride_i_b == stride_o_b);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
            RotateStyle,
            stride_i_d,
            kn_entry_1c_2d_cached_inplace<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1><<<grid, block, 0, stream>>>(
                p_output,
                p_cos_h, p_sin_h,
                p_cos_w, p_sin_w,
                img_width, size_h, size_d,
                stride_i_b, stride_i_s, stride_i_h, stride_i_d);
        );
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
            RotateStyle,
            stride_i_d,
            stride_o_d,
            kn_entry_1c_2d_cached<Op, RotateStyle, ReuseFreqsFrontPart, NopeFirst, Stride0Eq1, Stride1Eq1><<<grid, block, 0, stream>>>(
                p_output,
                p_input,
                p_cos_h, p_sin_h,
                p_cos_w, p_sin_w,
                img_width, size_h, size_d,
                stride_i_b, stride_i_s, stride_i_h, stride_i_d,
                stride_o_b, stride_o_s, stride_o_h, stride_o_d);
        );
    }
}

#define DISPATCH_ROPE_TYPES_PARAMS(TYPE0, TYPE1, ROTATE_STYLE, REUSE_FREQS_FRONT_PART, NOPE_FIRST, NAME, ...) \
    switch ((TYPE0)) {                                                                                        \
        case at::ScalarType::Float: {                                                                         \
            using scalar_t_0 = float;                                                                         \
            switch ((TYPE1))                                                                                  \
            {                                                                                                 \
                case at::ScalarType::Float: {                                                                 \
                    using scalar_t_1 = float;                                                                 \
                    if ((REUSE_FREQS_FRONT_PART))                                                             \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = true;                                            \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    else                                                                                      \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = false;                                           \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    break;                                                                                    \
                }                                                                                             \
                case at::ScalarType::Half: {                                                                  \
                    using scalar_t_1 = at::Half;                                                              \
                    if ((REUSE_FREQS_FRONT_PART))                                                             \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = true;                                            \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    else                                                                                      \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = false;                                           \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    break;                                                                                    \
                }                                                                                             \
                case at::ScalarType::BFloat16: {                                                              \
                    using scalar_t_1 = at::BFloat16;                                                          \
                    if ((REUSE_FREQS_FRONT_PART))                                                             \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = true;                                            \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    else                                                                                      \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = false;                                           \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    break;                                                                                    \
                }                                                                                             \
                default:                                                                                      \
                    TORCH_CHECK(false, NAME " does't support ",                                               \
                        toString((TYPE0)), " with ", toString((TYPE1)), ".");                                 \
            }                                                                                                 \
            break;                                                                                            \
        }                                                                                                     \
        case at::ScalarType::Half: {                                                                          \
            using scalar_t_0 = at::Half;                                                                      \
            switch ((TYPE1))                                                                                  \
            {                                                                                                 \
                case at::ScalarType::Float: {                                                                 \
                    using scalar_t_1 = float;                                                                 \
                    if ((REUSE_FREQS_FRONT_PART))                                                             \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = true;                                            \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    else                                                                                      \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = false;                                           \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    break;                                                                                    \
                }                                                                                             \
                case at::ScalarType::Half: {                                                                  \
                    using scalar_t_1 = at::Half;                                                              \
                    if ((REUSE_FREQS_FRONT_PART))                                                             \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = true;                                            \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    else                                                                                      \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = false;                                           \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    break;                                                                                    \
                }                                                                                             \
                case at::ScalarType::BFloat16: {                                                              \
                    using scalar_t_1 = at::BFloat16;                                                          \
                    if ((REUSE_FREQS_FRONT_PART))                                                             \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = true;                                            \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    else                                                                                      \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = false;                                           \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    break;                                                                                    \
                }                                                                                             \
                default:                                                                                      \
                    TORCH_CHECK(false, NAME " does't support ",                                               \
                        toString((TYPE0)), " with ", toString((TYPE1)), ".");                                 \
            }                                                                                                 \
            break;                                                                                            \
        }                                                                                                     \
        case at::ScalarType::BFloat16: {                                                                      \
            using scalar_t_0 = at::BFloat16;                                                                  \
            switch ((TYPE1))                                                                                  \
            {                                                                                                 \
                case at::ScalarType::Float: {                                                                 \
                    using scalar_t_1 = float;                                                                 \
                    if ((REUSE_FREQS_FRONT_PART))                                                             \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = true;                                            \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    else                                                                                      \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = false;                                           \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    break;                                                                                    \
                }                                                                                             \
                case at::ScalarType::Half: {                                                                  \
                    using scalar_t_1 = at::Half;                                                              \
                    if ((REUSE_FREQS_FRONT_PART))                                                             \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = true;                                            \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    else                                                                                      \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = false;                                           \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    break;                                                                                    \
                }                                                                                             \
                case at::ScalarType::BFloat16: {                                                              \
                    using scalar_t_1 = at::BFloat16;                                                          \
                    if ((REUSE_FREQS_FRONT_PART))                                                             \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = true;                                            \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    else                                                                                      \
                    {                                                                                         \
                        constexpr bool ReuseFreqsFrontPart = false;                                           \
                        if ((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                                              \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else if ((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                                         \
                        {                                                                                     \
                            constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;                                \
                            if ((NOPE_FIRST))                                                                 \
                            {                                                                                 \
                                constexpr bool NopeFirst = true;                                              \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                            else                                                                              \
                            {                                                                                 \
                                constexpr bool NopeFirst = false;                                             \
                                __VA_ARGS__;                                                                  \
                            }                                                                                 \
                        }                                                                                     \
                        else                                                                                  \
                        {                                                                                     \
                            TORCH_CHECK(false, NAME " does't support rotate type ",                           \
                                        std::to_string((ROTATE_STYLE)), ".");                                 \
                        }                                                                                     \
                    }                                                                                         \
                    break;                                                                                    \
                }                                                                                             \
                default:                                                                                      \
                    TORCH_CHECK(false, NAME " does't support ",                                               \
                        toString((TYPE0)), " with ", toString((TYPE1)), ".");                                 \
            }                                                                                                 \
            break;                                                                                            \
        }                                                                                                     \
        default:                                                                                              \
            TORCH_CHECK(false, NAME " does't support ", toString((TYPE0)), ".");                              \
    }

// =====================================================================================================================
// Interfaces
//

void rope_fwd_impl(
    torch::Tensor&       output,        // [s, b, h, d]
    const torch::Tensor& input,         // [s, b, h, d]
    const torch::Tensor& freqs,         // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
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
        nope_first,
        "dispatch_1c_sbhd_uncached<OpUncachedFwd, ...>",
        dispatch_1c_sbhd_uncached<OpUncachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
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
        nope_first,
        "dispatch_1c_sbhd_uncached<OpUncachedBwd, ...>",
        dispatch_1c_sbhd_uncached<OpUncachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    // Get sizes of input and output
    const int32_t size_s   = input_x.size(0);
    const int32_t size_b   = input_x.size(1);
    const int32_t size_h_x = input_x.size(2);
    const int32_t size_h_y = input_y.size(2);
    const int32_t size_d   = input_x.size(3);
    const int32_t size_f   = freqs.size(3);
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
        nope_first,
        "dispatch_2c_sbhd_uncached<OpUncachedFwd, ...>",
        dispatch_2c_sbhd_uncached<OpUncachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
            output_x.data_ptr<scalar_t_0>(),
            output_y.data_ptr<scalar_t_0>(),
            input_x.data_ptr<scalar_t_0>(),
            input_y.data_ptr<scalar_t_0>(),
            freqs.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h_x, size_h_y, size_d,
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    // Get sizes of input and output
    const int32_t size_s   = output_grads_x.size(0);
    const int32_t size_b   = output_grads_x.size(1);
    const int32_t size_h_x = output_grads_x.size(2);
    const int32_t size_h_y = output_grads_y.size(2);
    const int32_t size_d   = output_grads_x.size(3);
    const int32_t size_f   = freqs.size(3);
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
        nope_first,
        "dispatch_2c_sbhd_uncached<OpUncachedBwd, ...>",
        dispatch_2c_sbhd_uncached<OpUncachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
            input_grads_x.data_ptr<scalar_t_0>(),
            input_grads_y.data_ptr<scalar_t_0>(),
            output_grads_x.data_ptr<scalar_t_0>(),
            output_grads_y.data_ptr<scalar_t_0>(),
            freqs.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h_x, size_h_y, size_d,
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
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
        nope_first,
        "dispatch_1c_sbhd_cached<OpCachedFwd, ...>",
        dispatch_1c_sbhd_cached<OpCachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
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
        nope_first,
        "dispatch_1c_sbhd_cached<OpCachedBwd, ...>",
        dispatch_1c_sbhd_cached<OpCachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    // Get sizes of input and output
    const int32_t size_s   = input_x.size(0);
    const int32_t size_b   = input_x.size(1);
    const int32_t size_h_x = input_x.size(2);
    const int32_t size_h_y = input_y.size(2);
    const int32_t size_d   = input_x.size(3);
    const int32_t size_f   = cos.size(3);
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
        nope_first,
        "dispatch_2c_sbhd_cached<OpCachedFwd, ...>",
        dispatch_2c_sbhd_cached<OpCachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
            output_x.data_ptr<scalar_t_0>(),
            output_y.data_ptr<scalar_t_0>(),
            input_x.data_ptr<scalar_t_0>(),
            input_y.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h_x, size_h_y, size_d,
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    // Get sizes of input and output
    const int32_t size_s   = output_grads_x.size(0);
    const int32_t size_b   = output_grads_x.size(1);
    const int32_t size_h_x = output_grads_x.size(2);
    const int32_t size_h_y = output_grads_y.size(2);
    const int32_t size_d   = output_grads_x.size(3);
    const int32_t size_f   = cos.size(3);
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
        nope_first,
        "dispatch_2c_sbhd_cached<OpCachedBwd, ...>",
        dispatch_2c_sbhd_cached<OpCachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
            input_grads_x.data_ptr<scalar_t_0>(),
            input_grads_y.data_ptr<scalar_t_0>(),
            output_grads_x.data_ptr<scalar_t_0>(),
            output_grads_y.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h_x, size_h_y, size_d,
            size_f, // size of last dimension of freqs.
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d););
}

void rope_cached_positions_2c_fwd_impl(
    torch::Tensor&       output_x,      // [s, b, h, d]
    torch::Tensor&       output_y,      // [s, b, h, d]
    const torch::Tensor& input_x,       // [s, b, h, d]
    const torch::Tensor& input_y,       // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& sin,           // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& positions,     // [s, b]
    const int32_t        rotate_style,  // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    // Get sizes of input and output
    const int32_t size_s   = min(input_x.size(0), positions.size(0));
    const int32_t size_b   = min(input_x.size(1), positions.size(1));
    const int32_t size_h_x = input_x.size(2);
    const int32_t size_h_y = input_y.size(2);
    const int32_t size_d   = input_x.size(3);
    const int32_t size_f   = cos.size(3);
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
    // Get strides of positions and offsets
    assert(1 == positions.stride(1) && 2 == positions.dim());

    DISPATCH_ROPE_TYPES_PARAMS(
        input_x.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_2c_sbhd_cached_indirect<OpCachedFwd, ...>",
        dispatch_2c_sbhd_cached_indirect<OpCachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
            output_x.data_ptr<scalar_t_0>(),
            output_y.data_ptr<scalar_t_0>(),
            input_x.data_ptr<scalar_t_0>(),
            input_y.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            positions.data_ptr<int64_t>(),
            size_s, size_b, size_h_x, size_h_y, size_d,
            size_f, // size of last dimension of freqs.
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d););
}

void rope_cached_positions_offsets_2c_fwd_impl(
    torch::Tensor&       output_x,      // [s, b, h, d]
    torch::Tensor&       output_y,      // [s, b, h, d]
    const torch::Tensor& input_x,       // [s, b, h, d]
    const torch::Tensor& input_y,       // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& sin,           // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& positions,     // [s, b]
    const torch::Tensor& offsets,       // [s, b]
    const int32_t        rotate_style,  // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    // Get sizes of input and output
    const int32_t size_s   = min(min(input_x.size(0), positions.size(0)), offsets.size(0));
    const int32_t size_b   = min(min(input_x.size(1), positions.size(1)), offsets.size(1));
    const int32_t size_h_x = input_x.size(2);
    const int32_t size_h_y = input_y.size(2);
    const int32_t size_d   = input_x.size(3);
    const int32_t size_f   = cos.size(3);
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
    // Get strides of positions and offsets
    assert(1 == positions.stride(1) && 2 == positions.dim());
    assert(1 == offsets.stride(1)   && 2 == offsets.dim());

    DISPATCH_ROPE_TYPES_PARAMS(
        input_x.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_2c_sbhd_cached_indirect2<OpCachedFwd, ...>",
        dispatch_2c_sbhd_cached_indirect2<OpCachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
            output_x.data_ptr<scalar_t_0>(),
            output_y.data_ptr<scalar_t_0>(),
            input_x.data_ptr<scalar_t_0>(),
            input_y.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            positions.data_ptr<int64_t>(),
            offsets.data_ptr<int64_t>(),
            size_s, size_b, size_h_x, size_h_y, size_d,
            size_f, // size of last dimension of freqs.
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d););
}

void rope_thd_fwd_impl(
    torch::Tensor&       output,        // [t, h, d]
    const torch::Tensor& input,         // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [b + 1]
    const torch::Tensor& freqs,         // [max_s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
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
        nope_first,
        "dispatch_1c_thd_uncached<OpUncachedFwd, ...>",
        dispatch_1c_thd_uncached<OpUncachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
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
        nope_first,
        "dispatch_1c_thd_uncached<OpUncachedBwd, ...>",
        dispatch_1c_thd_uncached<OpUncachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
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
        nope_first,
        "dispatch_1c_2d_cached<OpCachedFwd, ...>",
        dispatch_1c_2d_cached<OpCachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
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
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
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
        nope_first,
        "dispatch_1c_2d_cached<OpCachedBwd, ...>",
        dispatch_1c_2d_cached<OpCachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
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
