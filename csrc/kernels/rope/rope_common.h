// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_hip_common.h"
#include "dispatch_utils.h"
#include "hip_float8.h"
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

// =====================================================================================================================
// Keyword interpretation
// ----------------------------------------------------------------
// 1c/2c:               The number of channels. 2c means two inputs and two outputs.
// Cached, Uncached:    Whether cosine and sine are calculated in kernel. Cached means kernel can
// read these value from
//                      memory rather than calculate these value according to the given theta in
//                      memory.
// ReuseFreqsFrontPart: Normally, freqs/cos/sin tensors should be repeated before conduct the RoPE
// operators. With
//                      this value set as true, the repeat is no longer required. Kernel can
//                      automatically relocate the desired element.
// sbhd:                Shape of tensor: [sequence length, batch size, head count, hidden
// dimension]. thd:                 Shape of tensor. 2d:                  2D image. NopeFirst: [0,
// size_r(rotate dim)) is rotated and the rest is just copied if this value is false.
//                      [size_d (size of d dim) - size_r, size_d) is rotated and the front part is
//                      just copied if true.
//

#define ROTATE_STYLE_NEOX 0
#define ROTATE_STYLE_GPTJ 1

namespace aiter {
// =====================================================================================================================
// Kernel Helper Functions
//

template <int32_t RotateStyle, bool IsForward, bool ReuseFreqsFrontPart, typename scalar_f_t>
__device__ __forceinline__ void load_cos_sin_uncached(float* p_cos_0,
                                                      float* p_sin_0,
                                                      float* p_cos_1,
                                                      float* p_sin_1,
                                                      const scalar_f_t* __restrict__ p_freqs,
                                                      const int32_t did,
                                                      const int32_t size_half_r)
{
    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        if constexpr(IsForward)
        {
            sincosf(float(p_freqs[did]), p_sin_0, p_cos_0);
            if constexpr(ReuseFreqsFrontPart)
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
            if constexpr(ReuseFreqsFrontPart)
            {
                *p_cos_0 = cosf(f_did_0);
                *p_sin_0 = sinf(f_did_0);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                const float f_did_1 = p_freqs[did + size_half_r];
                *p_cos_0            = cosf(f_did_0);
                *p_sin_0            = sinf(f_did_1);
                *p_cos_1            = cosf(f_did_1);
                *p_sin_1            = sinf(f_did_0);
            }
        }
    }
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        if constexpr(IsForward)
        {
            if constexpr(ReuseFreqsFrontPart)
            {
                sincosf(float(p_freqs[did]), p_sin_0, p_cos_0);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                sincosf(float(p_freqs[did * 2]), p_sin_0, p_cos_0);
                sincosf(float(p_freqs[did * 2 + 1]), p_sin_1, p_cos_1);
            }
        }
        else
        {
            if constexpr(ReuseFreqsFrontPart)
            {
                sincosf(float(p_freqs[did]), p_sin_0, p_cos_0);
                *p_cos_1 = *p_cos_0;
                *p_sin_1 = *p_sin_0;
            }
            else
            {
                const float f_did_0 = float(p_freqs[did * 2]);
                const float f_did_1 = float(p_freqs[did * 2 + 1]);
                *p_cos_0            = cosf(f_did_0);
                *p_sin_0            = sinf(f_did_1);
                *p_cos_1            = cosf(f_did_1);
                *p_sin_1            = sinf(f_did_0);
            }
        }
    }
}

template <int32_t RotateStyle, bool IsForward, bool ReuseFreqsFrontPart, typename scalar_f_t>
__device__ __forceinline__ void load_cos_sin_cached(float* p_cos_0,
                                                    float* p_sin_0,
                                                    float* p_cos_1,
                                                    float* p_sin_1,
                                                    const scalar_f_t* __restrict__ p_cos,
                                                    const scalar_f_t* __restrict__ p_sin,
                                                    const int32_t did,
                                                    const int32_t size_half_r)
{
    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        if constexpr(IsForward)
        {
            *p_cos_0 = float(p_cos[did]);
            *p_sin_0 = float(p_sin[did]);
            if constexpr(ReuseFreqsFrontPart)
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
            if constexpr(ReuseFreqsFrontPart)
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
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        if constexpr(IsForward)
        {
            if constexpr(ReuseFreqsFrontPart)
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
            if constexpr(ReuseFreqsFrontPart)
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
__device__ __forceinline__ void get_offset(int32_t* p_offset_0,
                                           int32_t* p_offset_1,
                                           const int32_t did,
                                           const int32_t hid,
                                           const int32_t stride_d,
                                           const int32_t stride_h,
                                           const int32_t size_half_r)
{
    const int32_t offset_h = hid * stride_h;

    if constexpr(RotateStyle == ROTATE_STYLE_NEOX)
    {
        *p_offset_0 = offset_h + did * stride_d;
        *p_offset_1 = *p_offset_0 + size_half_r * stride_d;
    }
    else if constexpr(RotateStyle == ROTATE_STYLE_GPTJ)
    {
        *p_offset_0 = offset_h + 2 * did * stride_d;
        if constexpr(StrideDEq1)
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
__device__ __forceinline__ void load_payload(o_scalar_t* p_data_0,
                                             o_scalar_t* p_data_1,
                                             const i_scalar_t* p_buffer,
                                             const int32_t did,
                                             const int32_t hid,
                                             const int32_t stride_d,
                                             const int32_t stride_h,
                                             const int32_t size_half_r)
{
    int32_t offset_0, offset_1;
    get_offset<RotateStyle, StrideDEq1>(
        &offset_0, &offset_1, did, hid, stride_d, stride_h, size_half_r);

    *p_data_0 = o_scalar_t(p_buffer[offset_0]);
    *p_data_1 = o_scalar_t(p_buffer[offset_1]);
}

template <int32_t RotateStyle, bool StrideDEq1, typename o_scalar_t, typename i_scalar_t>
__device__ __forceinline__ void store_payload(o_scalar_t* p_buffer,
                                              const i_scalar_t data_0,
                                              const i_scalar_t data_1,
                                              const int32_t did,
                                              const int32_t hid,
                                              const int32_t stride_d,
                                              const int32_t stride_h,
                                              const int32_t size_half_r)
{
    int32_t offset_0, offset_1;
    get_offset<RotateStyle, StrideDEq1>(
        &offset_0, &offset_1, did, hid, stride_d, stride_h, size_half_r);

    p_buffer[offset_0] = o_scalar_t(data_0);
    p_buffer[offset_1] = o_scalar_t(data_1);
}

template <typename scalar_t>
__device__ __forceinline__ void elementwise_copy(scalar_t* __restrict__ p_output,
                                                 const scalar_t* __restrict__ p_input,
                                                 const int32_t hid_end,
                                                 const int32_t did_start,
                                                 const int32_t did_end,
                                                 const int32_t stride_i_h,
                                                 const int32_t stride_i_d,
                                                 const int32_t stride_o_h,
                                                 const int32_t stride_o_d)
{
    if(did_end > did_start)
    {
        for(int32_t hid = threadIdx.y; hid < hid_end; hid += blockDim.y)
        {
            const int32_t offset_i = hid * stride_i_h;
            const int32_t offset_o = hid * stride_o_h;

            for(int32_t did = threadIdx.x + did_start; did < did_end; did += blockDim.x)
            {
                p_output[offset_o + did * stride_o_d] = p_input[offset_i + did * stride_i_d];
            }
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void elementwise_copy_2c(scalar_t* __restrict__ p_output_x,
                                                    scalar_t* __restrict__ p_output_y,
                                                    const scalar_t* __restrict__ p_input_x,
                                                    const scalar_t* __restrict__ p_input_y,
                                                    const int32_t hid_end_x,
                                                    const int32_t hid_end_y,
                                                    const int32_t did_start,
                                                    const int32_t did_end,
                                                    const int32_t stride_ix_h,
                                                    const int32_t stride_ix_d,
                                                    const int32_t stride_iy_h,
                                                    const int32_t stride_iy_d,
                                                    const int32_t stride_ox_h,
                                                    const int32_t stride_ox_d,
                                                    const int32_t stride_oy_h,
                                                    const int32_t stride_oy_d)
{
    if(did_end > did_start)
    {
        const int32_t hid_min_end = hid_end_x < hid_end_y ? hid_end_x : hid_end_y;

        for(int32_t hid = threadIdx.y; hid < hid_min_end; hid += blockDim.y)
        {
            const int32_t offset_ix = hid * stride_ix_h;
            const int32_t offset_iy = hid * stride_iy_h;
            const int32_t offset_ox = hid * stride_ox_h;
            const int32_t offset_oy = hid * stride_oy_h;

            for(int32_t did = threadIdx.x + did_start; did < did_end; did += blockDim.x)
            {
                p_output_x[offset_ox + did * stride_ox_d] =
                    p_input_x[offset_ix + did * stride_ix_d];
                p_output_y[offset_oy + did * stride_oy_d] =
                    p_input_y[offset_iy + did * stride_iy_d];
            }
        }

        for(int32_t hid = threadIdx.y + hid_min_end; hid < hid_end_x; hid += blockDim.y)
        {
            const int32_t offset_ix = hid * stride_ix_h;
            const int32_t offset_ox = hid * stride_ox_h;

            for(int32_t did = threadIdx.x + did_start; did < did_end; did += blockDim.x)
            {
                p_output_x[offset_ox + did * stride_ox_d] =
                    p_input_x[offset_ix + did * stride_ix_d];
            }
        }

        for(int32_t hid = threadIdx.y + hid_min_end; hid < hid_end_y; hid += blockDim.y)
        {
            const int32_t offset_iy = hid * stride_iy_h;
            const int32_t offset_oy = hid * stride_oy_h;

            for(int32_t did = threadIdx.x + did_start; did < did_end; did += blockDim.x)
            {
                p_output_y[offset_oy + did * stride_oy_d] =
                    p_input_y[offset_iy + did * stride_iy_d];
            }
        }
    }
}

// =====================================================================================================================
// Kernel Functionalities
//

struct OpUncachedFwd
{
    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDOutEq1,
              bool StrideDInEq1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void apply_1c(scalar_t* __restrict__ p_output,
                                                    const scalar_t* __restrict__ p_input,
                                                    const scalar_f_t* __restrict__ p_freqs,
                                                    const int32_t size_h,
                                                    const int32_t size_d,
                                                    const int32_t size_f,
                                                    const int32_t stride_i_h,
                                                    const int32_t stride_i_d,
                                                    const int32_t stride_o_h,
                                                    const int32_t stride_o_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;

        for(int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_uncached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did - did_start, size_half_r);

            for(int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
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
        if constexpr(!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy(p_output,
                             p_input,
                             size_h,
                             did_start,
                             did_end,
                             stride_i_h,
                             stride_i_d,
                             stride_o_h,
                             stride_o_d);
        }
    }

    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDOutXEq1,
              bool StrideDOutYEq1,
              bool StrideDInXEq1,
              bool StrideDInYEq1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void apply_2c(scalar_t* __restrict__ p_output_x,
                                                    scalar_t* __restrict__ p_output_y,
                                                    const scalar_t* __restrict__ p_input_x,
                                                    const scalar_t* __restrict__ p_input_y,
                                                    const scalar_f_t* __restrict__ p_freqs,
                                                    const int32_t size_h_x,
                                                    const int32_t size_h_y,
                                                    const int32_t size_d,
                                                    const int32_t size_f,
                                                    const int32_t stride_ix_h,
                                                    const int32_t stride_ix_d,
                                                    const int32_t stride_iy_h,
                                                    const int32_t stride_iy_d,
                                                    const int32_t stride_ox_h,
                                                    const int32_t stride_ox_d,
                                                    const int32_t stride_oy_h,
                                                    const int32_t stride_oy_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);

        for(int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_uncached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did - did_start, size_half_r);

            for(int32_t hid = threadIdx.y; hid < size_min_h; hid += blockDim.y)
            {
                float input_x_0, input_x_1, input_y_0, input_y_1;
                load_payload<RotateStyle, StrideDInXEq1>(&input_x_0,
                                                         &input_x_1,
                                                         p_input_x,
                                                         did,
                                                         hid,
                                                         stride_ix_d,
                                                         stride_ix_h,
                                                         size_half_r);
                load_payload<RotateStyle, StrideDInYEq1>(&input_y_0,
                                                         &input_y_1,
                                                         p_input_y,
                                                         did,
                                                         hid,
                                                         stride_iy_d,
                                                         stride_iy_h,
                                                         size_half_r);

                const float output_x_0 = input_x_0 * cos_0 - input_x_1 * sin_0;
                const float output_x_1 = input_x_1 * cos_1 + input_x_0 * sin_1;
                const float output_y_0 = input_y_0 * cos_0 - input_y_1 * sin_0;
                const float output_y_1 = input_y_1 * cos_1 + input_y_0 * sin_1;
                store_payload<RotateStyle, StrideDOutXEq1>(p_output_x,
                                                           output_x_0,
                                                           output_x_1,
                                                           did,
                                                           hid,
                                                           stride_ox_d,
                                                           stride_ox_h,
                                                           size_half_r);
                store_payload<RotateStyle, StrideDOutYEq1>(p_output_y,
                                                           output_y_0,
                                                           output_y_1,
                                                           did,
                                                           hid,
                                                           stride_oy_d,
                                                           stride_oy_h,
                                                           size_half_r);
            }

            for(int32_t hid = threadIdx.y + size_min_h; hid < size_h_x; hid += blockDim.y)
            {
                float input_x_0, input_x_1;
                load_payload<RotateStyle, StrideDInXEq1>(&input_x_0,
                                                         &input_x_1,
                                                         p_input_x,
                                                         did,
                                                         hid,
                                                         stride_ix_d,
                                                         stride_ix_h,
                                                         size_half_r);

                const float output_x_0 = input_x_0 * cos_0 - input_x_1 * sin_0;
                const float output_x_1 = input_x_1 * cos_1 + input_x_0 * sin_1;
                store_payload<RotateStyle, StrideDOutXEq1>(p_output_x,
                                                           output_x_0,
                                                           output_x_1,
                                                           did,
                                                           hid,
                                                           stride_ox_d,
                                                           stride_ox_h,
                                                           size_half_r);
            }

            for(int32_t hid = threadIdx.y + size_min_h; hid < size_h_y; hid += blockDim.y)
            {
                float input_y_0, input_y_1;
                load_payload<RotateStyle, StrideDInYEq1>(&input_y_0,
                                                         &input_y_1,
                                                         p_input_y,
                                                         did,
                                                         hid,
                                                         stride_iy_d,
                                                         stride_iy_h,
                                                         size_half_r);

                const float output_y_0 = input_y_0 * cos_0 - input_y_1 * sin_0;
                const float output_y_1 = input_y_1 * cos_1 + input_y_0 * sin_1;
                store_payload<RotateStyle, StrideDOutYEq1>(p_output_y,
                                                           output_y_0,
                                                           output_y_1,
                                                           did,
                                                           hid,
                                                           stride_oy_d,
                                                           stride_oy_h,
                                                           size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr(!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c(p_output_x,
                                p_output_y,
                                p_input_x,
                                p_input_y,
                                size_h_x,
                                size_h_y,
                                did_start,
                                did_end,
                                stride_ix_h,
                                stride_ix_d,
                                stride_iy_h,
                                stride_iy_d,
                                stride_ox_h,
                                stride_ox_d,
                                stride_oy_h,
                                stride_oy_d);
        }
    }
};

struct OpUncachedBwd
{
    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDInGradsEq1,
              bool StrideDOutGradsEq1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void apply_1c(scalar_t* __restrict__ p_input_grads,
                                                    const scalar_t* __restrict__ p_output_grads,
                                                    const scalar_f_t* __restrict__ p_freqs,
                                                    const int32_t size_h,
                                                    const int32_t size_d,
                                                    const int32_t size_f,
                                                    const int32_t stride_o_h,
                                                    const int32_t stride_o_d,
                                                    const int32_t stride_i_h,
                                                    const int32_t stride_i_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;

        for(int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_uncached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did - did_start, size_half_r);

            for(int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                float output_grad_0, output_grad_1;
                load_payload<RotateStyle, StrideDOutGradsEq1>(&output_grad_0,
                                                              &output_grad_1,
                                                              p_output_grads,
                                                              did,
                                                              hid,
                                                              stride_o_d,
                                                              stride_o_h,
                                                              size_half_r);

                const float input_grad_0 = output_grad_0 * cos_0 + output_grad_1 * sin_0;
                const float input_grad_1 = output_grad_1 * cos_1 - output_grad_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsEq1>(p_input_grads,
                                                              input_grad_0,
                                                              input_grad_1,
                                                              did,
                                                              hid,
                                                              stride_i_d,
                                                              stride_i_h,
                                                              size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr(!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy(p_input_grads,
                             p_output_grads,
                             size_h,
                             did_start,
                             did_end,
                             stride_o_h,
                             stride_o_d,
                             stride_i_h,
                             stride_i_d);
        }
    }

    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDInGradsXEq1,
              bool StrideDInGradsYEq1,
              bool StrideDOutGradsXEq1,
              bool StrideDOutGradsYEq1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void apply_2c(scalar_t* __restrict__ p_input_grads_x,
                                                    scalar_t* __restrict__ p_input_grads_y,
                                                    const scalar_t* __restrict__ p_output_grads_x,
                                                    const scalar_t* __restrict__ p_output_grads_y,
                                                    const scalar_f_t* __restrict__ p_freqs,
                                                    const int32_t size_h_x,
                                                    const int32_t size_h_y,
                                                    const int32_t size_d,
                                                    const int32_t size_f,
                                                    const int32_t stride_ox_h,
                                                    const int32_t stride_ox_d,
                                                    const int32_t stride_oy_h,
                                                    const int32_t stride_oy_d,
                                                    const int32_t stride_ix_h,
                                                    const int32_t stride_ix_d,
                                                    const int32_t stride_iy_h,
                                                    const int32_t stride_iy_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);

        for(int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_uncached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_freqs, did - did_start, size_half_r);

            for(int32_t hid = threadIdx.y; hid < size_min_h; hid += blockDim.y)
            {
                float output_grad_x_0, output_grad_x_1, output_grad_y_0, output_grad_y_1;
                load_payload<RotateStyle, StrideDOutGradsXEq1>(&output_grad_x_0,
                                                               &output_grad_x_1,
                                                               p_output_grads_x,
                                                               did,
                                                               hid,
                                                               stride_ox_d,
                                                               stride_ox_h,
                                                               size_half_r);
                load_payload<RotateStyle, StrideDOutGradsYEq1>(&output_grad_y_0,
                                                               &output_grad_y_1,
                                                               p_output_grads_y,
                                                               did,
                                                               hid,
                                                               stride_oy_d,
                                                               stride_oy_h,
                                                               size_half_r);

                const float input_grad_x_0 = output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0;
                const float input_grad_x_1 = output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1;
                const float input_grad_y_0 = output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0;
                const float input_grad_y_1 = output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsXEq1>(p_input_grads_x,
                                                               input_grad_x_0,
                                                               input_grad_x_1,
                                                               did,
                                                               hid,
                                                               stride_ix_d,
                                                               stride_ix_h,
                                                               size_half_r);
                store_payload<RotateStyle, StrideDInGradsYEq1>(p_input_grads_y,
                                                               input_grad_y_0,
                                                               input_grad_y_1,
                                                               did,
                                                               hid,
                                                               stride_iy_d,
                                                               stride_iy_h,
                                                               size_half_r);
            }

            for(int32_t hid = threadIdx.y + size_min_h; hid < size_h_x; hid += blockDim.y)
            {
                float output_grad_x_0, output_grad_x_1;
                load_payload<RotateStyle, StrideDOutGradsXEq1>(&output_grad_x_0,
                                                               &output_grad_x_1,
                                                               p_output_grads_x,
                                                               did,
                                                               hid,
                                                               stride_ox_d,
                                                               stride_ox_h,
                                                               size_half_r);

                const float input_grad_x_0 = output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0;
                const float input_grad_x_1 = output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsXEq1>(p_input_grads_x,
                                                               input_grad_x_0,
                                                               input_grad_x_1,
                                                               did,
                                                               hid,
                                                               stride_ix_d,
                                                               stride_ix_h,
                                                               size_half_r);
            }

            for(int32_t hid = threadIdx.y + size_min_h; hid < size_h_y; hid += blockDim.y)
            {
                float output_grad_y_0, output_grad_y_1;
                load_payload<RotateStyle, StrideDOutGradsYEq1>(&output_grad_y_0,
                                                               &output_grad_y_1,
                                                               p_output_grads_y,
                                                               did,
                                                               hid,
                                                               stride_oy_d,
                                                               stride_oy_h,
                                                               size_half_r);

                const float input_grad_y_0 = output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0;
                const float input_grad_y_1 = output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsYEq1>(p_input_grads_y,
                                                               input_grad_y_0,
                                                               input_grad_y_1,
                                                               did,
                                                               hid,
                                                               stride_iy_d,
                                                               stride_iy_h,
                                                               size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr(!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c(p_input_grads_x,
                                p_input_grads_y,
                                p_output_grads_x,
                                p_output_grads_y,
                                size_h_x,
                                size_h_y,
                                did_start,
                                did_end,
                                stride_ox_h,
                                stride_ox_d,
                                stride_oy_h,
                                stride_oy_d,
                                stride_ix_h,
                                stride_ix_d,
                                stride_iy_h,
                                stride_iy_d);
        }
    }
};

struct OpCachedFwd
{
    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDOutEq1,
              bool StrideDInEq1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void apply_1c(scalar_t* __restrict__ p_output,
                                                    const scalar_t* __restrict__ p_input,
                                                    const scalar_f_t* __restrict__ p_cos,
                                                    const scalar_f_t* __restrict__ p_sin,
                                                    const int32_t size_h,
                                                    const int32_t size_d,
                                                    const int32_t size_f,
                                                    const int32_t stride_i_h,
                                                    const int32_t stride_i_d,
                                                    const int32_t stride_o_h,
                                                    const int32_t stride_o_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;

        for(int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_cached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did - did_start, size_half_r);

            for(int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
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
        if constexpr(!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy(p_output,
                             p_input,
                             size_h,
                             did_start,
                             did_end,
                             stride_i_h,
                             stride_i_d,
                             stride_o_h,
                             stride_o_d);
        }
    }

    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDOutXEq1,
              bool StrideDOutYEq1,
              bool StrideDInXEq1,
              bool StrideDInYEq1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void apply_2c(scalar_t* __restrict__ p_output_x,
                                                    scalar_t* __restrict__ p_output_y,
                                                    const scalar_t* __restrict__ p_input_x,
                                                    const scalar_t* __restrict__ p_input_y,
                                                    const scalar_f_t* __restrict__ p_cos,
                                                    const scalar_f_t* __restrict__ p_sin,
                                                    const int32_t size_h_x,
                                                    const int32_t size_h_y,
                                                    const int32_t size_d,
                                                    const int32_t size_f,
                                                    const int32_t stride_ix_h,
                                                    const int32_t stride_ix_d,
                                                    const int32_t stride_iy_h,
                                                    const int32_t stride_iy_d,
                                                    const int32_t stride_ox_h,
                                                    const int32_t stride_ox_d,
                                                    const int32_t stride_oy_h,
                                                    const int32_t stride_oy_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);

        for(int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_cached<RotateStyle, true, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did - did_start, size_half_r);

            for(int32_t hid = threadIdx.y; hid < size_min_h; hid += blockDim.y)
            {
                float input_x_0, input_x_1, input_y_0, input_y_1;
                load_payload<RotateStyle, StrideDInXEq1>(&input_x_0,
                                                         &input_x_1,
                                                         p_input_x,
                                                         did,
                                                         hid,
                                                         stride_ix_d,
                                                         stride_ix_h,
                                                         size_half_r);
                load_payload<RotateStyle, StrideDInYEq1>(&input_y_0,
                                                         &input_y_1,
                                                         p_input_y,
                                                         did,
                                                         hid,
                                                         stride_iy_d,
                                                         stride_iy_h,
                                                         size_half_r);

                const float output_x_0 = input_x_0 * cos_0 - input_x_1 * sin_0;
                const float output_x_1 = input_x_1 * cos_1 + input_x_0 * sin_1;
                const float output_y_0 = input_y_0 * cos_0 - input_y_1 * sin_0;
                const float output_y_1 = input_y_1 * cos_1 + input_y_0 * sin_1;
                store_payload<RotateStyle, StrideDOutXEq1>(p_output_x,
                                                           output_x_0,
                                                           output_x_1,
                                                           did,
                                                           hid,
                                                           stride_ox_d,
                                                           stride_ox_h,
                                                           size_half_r);
                store_payload<RotateStyle, StrideDOutYEq1>(p_output_y,
                                                           output_y_0,
                                                           output_y_1,
                                                           did,
                                                           hid,
                                                           stride_oy_d,
                                                           stride_oy_h,
                                                           size_half_r);
            }

            for(int32_t hid = threadIdx.y + size_min_h; hid < size_h_x; hid += blockDim.y)
            {
                float input_x_0, input_x_1;
                load_payload<RotateStyle, StrideDInXEq1>(&input_x_0,
                                                         &input_x_1,
                                                         p_input_x,
                                                         did,
                                                         hid,
                                                         stride_ix_d,
                                                         stride_ix_h,
                                                         size_half_r);

                const float output_x_0 = input_x_0 * cos_0 - input_x_1 * sin_0;
                const float output_x_1 = input_x_1 * cos_1 + input_x_0 * sin_1;
                store_payload<RotateStyle, StrideDOutXEq1>(p_output_x,
                                                           output_x_0,
                                                           output_x_1,
                                                           did,
                                                           hid,
                                                           stride_ox_d,
                                                           stride_ox_h,
                                                           size_half_r);
            }

            for(int32_t hid = threadIdx.y + size_min_h; hid < size_h_y; hid += blockDim.y)
            {
                float input_y_0, input_y_1;
                load_payload<RotateStyle, StrideDInYEq1>(&input_y_0,
                                                         &input_y_1,
                                                         p_input_y,
                                                         did,
                                                         hid,
                                                         stride_iy_d,
                                                         stride_iy_h,
                                                         size_half_r);

                const float output_y_0 = input_y_0 * cos_0 - input_y_1 * sin_0;
                const float output_y_1 = input_y_1 * cos_1 + input_y_0 * sin_1;
                store_payload<RotateStyle, StrideDOutYEq1>(p_output_y,
                                                           output_y_0,
                                                           output_y_1,
                                                           did,
                                                           hid,
                                                           stride_oy_d,
                                                           stride_oy_h,
                                                           size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr(!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c(p_output_x,
                                p_output_y,
                                p_input_x,
                                p_input_y,
                                size_h_x,
                                size_h_y,
                                did_start,
                                did_end,
                                stride_ix_h,
                                stride_ix_d,
                                stride_iy_h,
                                stride_iy_d,
                                stride_ox_h,
                                stride_ox_d,
                                stride_oy_h,
                                stride_oy_d);
        }
    }
};

struct OpCachedBwd
{
    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDInGradsEq1,
              bool StrideDOutGradsEq1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void apply_1c(scalar_t* __restrict__ p_input_grads,
                                                    const scalar_t* __restrict__ p_output_grads,
                                                    const scalar_f_t* __restrict__ p_cos,
                                                    const scalar_f_t* __restrict__ p_sin,
                                                    const int32_t size_h,
                                                    const int32_t size_d,
                                                    const int32_t size_f,
                                                    const int32_t stride_o_h,
                                                    const int32_t stride_o_d,
                                                    const int32_t stride_i_h,
                                                    const int32_t stride_i_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;

        for(int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_cached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did - did_start, size_half_r);

            for(int32_t hid = threadIdx.y; hid < size_h; hid += blockDim.y)
            {
                float output_grad_0, output_grad_1;
                load_payload<RotateStyle, StrideDOutGradsEq1>(&output_grad_0,
                                                              &output_grad_1,
                                                              p_output_grads,
                                                              did,
                                                              hid,
                                                              stride_o_d,
                                                              stride_o_h,
                                                              size_half_r);

                const float input_grad_0 = output_grad_0 * cos_0 + output_grad_1 * sin_0;
                const float input_grad_1 = output_grad_1 * cos_1 - output_grad_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsEq1>(p_input_grads,
                                                              input_grad_0,
                                                              input_grad_1,
                                                              did,
                                                              hid,
                                                              stride_i_d,
                                                              stride_i_h,
                                                              size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr(!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy(p_input_grads,
                             p_output_grads,
                             size_h,
                             did_start,
                             did_end,
                             stride_o_h,
                             stride_o_d,
                             stride_i_h,
                             stride_i_d);
        }
    }

    template <int32_t RotateStyle,
              bool ReuseFreqsFrontPart,
              bool NopeFirst,
              bool Inplace,
              bool StrideDInGradsXEq1,
              bool StrideDInGradsYEq1,
              bool StrideDOutGradsXEq1,
              bool StrideDOutGradsYEq1,
              typename scalar_t,
              typename scalar_f_t>
    __device__ __forceinline__ static void apply_2c(scalar_t* __restrict__ p_input_grads_x,
                                                    scalar_t* __restrict__ p_input_grads_y,
                                                    const scalar_t* __restrict__ p_output_grads_x,
                                                    const scalar_t* __restrict__ p_output_grads_y,
                                                    const scalar_f_t* __restrict__ p_cos,
                                                    const scalar_f_t* __restrict__ p_sin,
                                                    const int32_t size_h_x,
                                                    const int32_t size_h_y,
                                                    const int32_t size_d,
                                                    const int32_t size_f,
                                                    const int32_t stride_ox_h,
                                                    const int32_t stride_ox_d,
                                                    const int32_t stride_oy_h,
                                                    const int32_t stride_oy_d,
                                                    const int32_t stride_ix_h,
                                                    const int32_t stride_ix_d,
                                                    const int32_t stride_iy_h,
                                                    const int32_t stride_iy_d)
    {
        // rotate count
        const int32_t size_r      = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
        const int32_t size_half_r = size_r >> 1;
        const int32_t did_start   = NopeFirst ? (size_d - size_r) : 0;
        const int32_t size_min_h  = min(size_h_x, size_h_y);

        for(int32_t did = threadIdx.x + did_start; did < size_half_r; did += blockDim.x)
        {
            float cos_0, sin_0, cos_1, sin_1;
            load_cos_sin_cached<RotateStyle, false, ReuseFreqsFrontPart>(
                &cos_0, &sin_0, &cos_1, &sin_1, p_cos, p_sin, did - did_start, size_half_r);

            for(int32_t hid = threadIdx.y; hid < size_min_h; hid += blockDim.y)
            {
                float output_grad_x_0, output_grad_x_1, output_grad_y_0, output_grad_y_1;
                load_payload<RotateStyle, StrideDOutGradsXEq1>(&output_grad_x_0,
                                                               &output_grad_x_1,
                                                               p_output_grads_x,
                                                               did,
                                                               hid,
                                                               stride_ox_d,
                                                               stride_ox_h,
                                                               size_half_r);
                load_payload<RotateStyle, StrideDOutGradsYEq1>(&output_grad_y_0,
                                                               &output_grad_y_1,
                                                               p_output_grads_y,
                                                               did,
                                                               hid,
                                                               stride_oy_d,
                                                               stride_oy_h,
                                                               size_half_r);

                const float input_grad_x_0 = output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0;
                const float input_grad_x_1 = output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1;
                const float input_grad_y_0 = output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0;
                const float input_grad_y_1 = output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsXEq1>(p_input_grads_x,
                                                               input_grad_x_0,
                                                               input_grad_x_1,
                                                               did,
                                                               hid,
                                                               stride_ix_d,
                                                               stride_ix_h,
                                                               size_half_r);
                store_payload<RotateStyle, StrideDInGradsYEq1>(p_input_grads_y,
                                                               input_grad_y_0,
                                                               input_grad_y_1,
                                                               did,
                                                               hid,
                                                               stride_iy_d,
                                                               stride_iy_h,
                                                               size_half_r);
            }

            for(int32_t hid = threadIdx.y + size_min_h; hid < size_h_x; hid += blockDim.y)
            {
                float output_grad_x_0, output_grad_x_1;
                load_payload<RotateStyle, StrideDOutGradsXEq1>(&output_grad_x_0,
                                                               &output_grad_x_1,
                                                               p_output_grads_x,
                                                               did,
                                                               hid,
                                                               stride_ox_d,
                                                               stride_ox_h,
                                                               size_half_r);

                const float input_grad_x_0 = output_grad_x_0 * cos_0 + output_grad_x_1 * sin_0;
                const float input_grad_x_1 = output_grad_x_1 * cos_1 - output_grad_x_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsXEq1>(p_input_grads_x,
                                                               input_grad_x_0,
                                                               input_grad_x_1,
                                                               did,
                                                               hid,
                                                               stride_ix_d,
                                                               stride_ix_h,
                                                               size_half_r);
            }

            for(int32_t hid = threadIdx.y + size_min_h; hid < size_h_y; hid += blockDim.y)
            {
                float output_grad_y_0, output_grad_y_1;
                load_payload<RotateStyle, StrideDOutGradsYEq1>(&output_grad_y_0,
                                                               &output_grad_y_1,
                                                               p_output_grads_y,
                                                               did,
                                                               hid,
                                                               stride_oy_d,
                                                               stride_oy_h,
                                                               size_half_r);

                const float input_grad_y_0 = output_grad_y_0 * cos_0 + output_grad_y_1 * sin_0;
                const float input_grad_y_1 = output_grad_y_1 * cos_1 - output_grad_y_0 * sin_1;
                store_payload<RotateStyle, StrideDInGradsYEq1>(p_input_grads_y,
                                                               input_grad_y_0,
                                                               input_grad_y_1,
                                                               did,
                                                               hid,
                                                               stride_iy_d,
                                                               stride_iy_h,
                                                               size_half_r);
            }
        }

        // the rest are just forwarded
        if constexpr(!Inplace)
        {
            const int32_t did_start = NopeFirst ? 0 : size_r;
            const int32_t did_end   = NopeFirst ? (size_d - size_r) : size_d;
            elementwise_copy_2c(p_input_grads_x,
                                p_input_grads_y,
                                p_output_grads_x,
                                p_output_grads_y,
                                size_h_x,
                                size_h_y,
                                did_start,
                                did_end,
                                stride_ox_h,
                                stride_ox_d,
                                stride_oy_h,
                                stride_oy_d,
                                stride_ix_h,
                                stride_ix_d,
                                stride_iy_h,
                                stride_iy_d);
        }
    }
};

// =====================================================================================================================
// Kernel Entries
//

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_uncached(scalar_t* __restrict__ p_output,
                                   const scalar_t* __restrict__ p_input,
                                   const scalar_f_t* __restrict__ p_freqs,
                                   const int32_t size_h,
                                   const int32_t size_d,
                                   const int32_t size_f, // size of last dimension of freqs.
                                   const int32_t stride_i_s,
                                   const int32_t stride_i_b,
                                   const int32_t stride_i_h,
                                   const int32_t stride_i_d,
                                   const int32_t stride_o_s,
                                   const int32_t stride_o_b,
                                   const int32_t stride_o_h,
                                   const int32_t stride_o_d)
{
    const uint64_t sid      = blockIdx.x;
    const uint64_t bid      = blockIdx.y;
    const uint64_t offset_i = sid * stride_i_s + bid * stride_i_b;
    const uint64_t offset_o = sid * stride_o_s + bid * stride_o_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutEq1,
                          StrideDInEq1>(p_output + offset_o,
                                        p_input + offset_i,
                                        p_freqs + offset_f,
                                        size_h,
                                        size_d,
                                        size_f,
                                        stride_i_h,
                                        stride_i_d,
                                        stride_o_h,
                                        stride_o_d);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_uncached_inplace(scalar_t* __restrict__ p_inout,
                                           const scalar_f_t* __restrict__ p_freqs,
                                           const int32_t size_h,
                                           const int32_t size_d,
                                           const int32_t size_f, // size of last dimension of freqs.
                                           const int32_t stride_s,
                                           const int32_t stride_b,
                                           const int32_t stride_h,
                                           const int32_t stride_d)
{
    const uint64_t sid      = blockIdx.x;
    const uint64_t bid      = blockIdx.y;
    const uint64_t offset   = sid * stride_s + bid * stride_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDEq1,
                          StrideDEq1>(p_inout + offset,
                                      p_inout + offset,
                                      p_freqs + offset_f,
                                      size_h,
                                      size_d,
                                      size_f,
                                      stride_h,
                                      stride_d,
                                      stride_h,
                                      stride_d);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutXEq1,
          bool StrideDOutYEq1,
          bool StrideDInXEq1,
          bool StrideDInYEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_uncached(scalar_t* __restrict__ p_output_x,
                                   scalar_t* __restrict__ p_output_y,
                                   const scalar_t* __restrict__ p_input_x,
                                   const scalar_t* __restrict__ p_input_y,
                                   const scalar_f_t* __restrict__ p_freqs,
                                   const int32_t size_h_x,
                                   const int32_t size_h_y,
                                   const int32_t size_d,
                                   const int32_t size_f, // size of last dimension of freqs.
                                   const int32_t stride_ix_s,
                                   const int32_t stride_ix_b,
                                   const int32_t stride_ix_h,
                                   const int32_t stride_ix_d,
                                   const int32_t stride_iy_s,
                                   const int32_t stride_iy_b,
                                   const int32_t stride_iy_h,
                                   const int32_t stride_iy_d,
                                   const int32_t stride_ox_s,
                                   const int32_t stride_ox_b,
                                   const int32_t stride_ox_h,
                                   const int32_t stride_ox_d,
                                   const int32_t stride_oy_s,
                                   const int32_t stride_oy_b,
                                   const int32_t stride_oy_h,
                                   const int32_t stride_oy_d)
{
    const uint64_t sid       = blockIdx.x;
    const uint64_t bid       = blockIdx.y;
    const uint64_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const uint64_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const uint64_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const uint64_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const uint64_t offset_f  = sid * size_f;

    Op::template apply_2c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutXEq1,
                          StrideDOutYEq1,
                          StrideDInXEq1,
                          StrideDInYEq1>(p_output_x + offset_ox,
                                         p_output_y + offset_oy,
                                         p_input_x + offset_ix,
                                         p_input_y + offset_iy,
                                         p_freqs + offset_f,
                                         size_h_x,
                                         size_h_y,
                                         size_d,
                                         size_f,
                                         stride_ix_h,
                                         stride_ix_d,
                                         stride_iy_h,
                                         stride_iy_d,
                                         stride_ox_h,
                                         stride_ox_d,
                                         stride_oy_h,
                                         stride_oy_d);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDXEq1,
          bool StrideDYEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_uncached_inplace(scalar_t* __restrict__ p_inout_x,
                                           scalar_t* __restrict__ p_inout_y,
                                           const scalar_f_t* __restrict__ p_freqs,
                                           const int32_t size_h_x,
                                           const int32_t size_h_y,
                                           const int32_t size_d,
                                           const int32_t size_f, // size of last dimension of freqs.
                                           const int32_t stride_x_s,
                                           const int32_t stride_x_b,
                                           const int32_t stride_x_h,
                                           const int32_t stride_x_d,
                                           const int32_t stride_y_s,
                                           const int32_t stride_y_b,
                                           const int32_t stride_y_h,
                                           const int32_t stride_y_d)
{
    const uint64_t sid      = blockIdx.x;
    const uint64_t bid      = blockIdx.y;
    const uint64_t offset_x = sid * stride_x_s + bid * stride_x_b;
    const uint64_t offset_y = sid * stride_y_s + bid * stride_y_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_2c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDXEq1,
                          StrideDYEq1,
                          StrideDXEq1,
                          StrideDYEq1>(p_inout_x + offset_x,
                                       p_inout_y + offset_y,
                                       p_inout_x + offset_x,
                                       p_inout_y + offset_y,
                                       p_freqs + offset_f,
                                       size_h_x,
                                       size_h_y,
                                       size_d,
                                       size_f,
                                       stride_x_h,
                                       stride_x_d,
                                       stride_y_h,
                                       stride_y_d,
                                       stride_x_h,
                                       stride_x_d,
                                       stride_y_h,
                                       stride_y_d);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_cached(scalar_t* __restrict__ p_output,
                                 const scalar_t* __restrict__ p_input,
                                 const scalar_f_t* __restrict__ p_cos,
                                 const scalar_f_t* __restrict__ p_sin,
                                 const int32_t size_h,
                                 const int32_t size_d,
                                 const int32_t size_f, // size of last dimension of freqs.
                                 const int32_t stride_i_s,
                                 const int32_t stride_i_b,
                                 const int32_t stride_i_h,
                                 const int32_t stride_i_d,
                                 const int32_t stride_o_s,
                                 const int32_t stride_o_b,
                                 const int32_t stride_o_h,
                                 const int32_t stride_o_d)
{
    const uint64_t sid      = blockIdx.x;
    const uint64_t bid      = blockIdx.y;
    const uint64_t offset_i = sid * stride_i_s + bid * stride_i_b;
    const uint64_t offset_o = sid * stride_o_s + bid * stride_o_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutEq1,
                          StrideDInEq1>(p_output + offset_o,
                                        p_input + offset_i,
                                        p_cos + offset_f,
                                        p_sin + offset_f,
                                        size_h,
                                        size_d,
                                        size_f,
                                        stride_i_h,
                                        stride_i_d,
                                        stride_o_h,
                                        stride_o_d);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_cached_inplace(scalar_t* __restrict__ p_inout,
                                         const scalar_f_t* __restrict__ p_cos,
                                         const scalar_f_t* __restrict__ p_sin,
                                         const int32_t size_h,
                                         const int32_t size_d,
                                         const int32_t size_f, // size of last dimension of freqs.
                                         const int32_t stride_s,
                                         const int32_t stride_b,
                                         const int32_t stride_h,
                                         const int32_t stride_d)
{
    const uint64_t sid      = blockIdx.x;
    const uint64_t bid      = blockIdx.y;
    const uint64_t offset   = sid * stride_s + bid * stride_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDEq1,
                          StrideDEq1>(p_inout + offset,
                                      p_inout + offset,
                                      p_cos + offset_f,
                                      p_sin + offset_f,
                                      size_h,
                                      size_d,
                                      size_f,
                                      stride_h,
                                      stride_d,
                                      stride_h,
                                      stride_d);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutXEq1,
          bool StrideDOutYEq1,
          bool StrideDInXEq1,
          bool StrideDInYEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_cached(scalar_t* __restrict__ p_output_x,
                                 scalar_t* __restrict__ p_output_y,
                                 const scalar_t* __restrict__ p_input_x,
                                 const scalar_t* __restrict__ p_input_y,
                                 const scalar_f_t* __restrict__ p_cos,
                                 const scalar_f_t* __restrict__ p_sin,
                                 const int32_t size_h_x,
                                 const int32_t size_h_y,
                                 const int32_t size_d,
                                 const int32_t size_f, // size of last dimension of freqs.
                                 const int32_t stride_ix_s,
                                 const int32_t stride_ix_b,
                                 const int32_t stride_ix_h,
                                 const int32_t stride_ix_d,
                                 const int32_t stride_iy_s,
                                 const int32_t stride_iy_b,
                                 const int32_t stride_iy_h,
                                 const int32_t stride_iy_d,
                                 const int32_t stride_ox_s,
                                 const int32_t stride_ox_b,
                                 const int32_t stride_ox_h,
                                 const int32_t stride_ox_d,
                                 const int32_t stride_oy_s,
                                 const int32_t stride_oy_b,
                                 const int32_t stride_oy_h,
                                 const int32_t stride_oy_d)
{
    const uint64_t sid       = blockIdx.x;
    const uint64_t bid       = blockIdx.y;
    const uint64_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
    const uint64_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
    const uint64_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
    const uint64_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
    const uint64_t offset_f  = sid * size_f;

    Op::template apply_2c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutXEq1,
                          StrideDOutYEq1,
                          StrideDInXEq1,
                          StrideDInYEq1>(p_output_x + offset_ox,
                                         p_output_y + offset_oy,
                                         p_input_x + offset_ix,
                                         p_input_y + offset_iy,
                                         p_cos + offset_f,
                                         p_sin + offset_f,
                                         size_h_x,
                                         size_h_y,
                                         size_d,
                                         size_f,
                                         stride_ix_h,
                                         stride_ix_d,
                                         stride_iy_h,
                                         stride_iy_d,
                                         stride_ox_h,
                                         stride_ox_d,
                                         stride_oy_h,
                                         stride_oy_d);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDXEq1,
          bool StrideDYEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_cached_inplace(scalar_t* __restrict__ p_inout_x,
                                         scalar_t* __restrict__ p_inout_y,
                                         const scalar_f_t* __restrict__ p_cos,
                                         const scalar_f_t* __restrict__ p_sin,
                                         const int32_t size_h_x,
                                         const int32_t size_h_y,
                                         const int32_t size_d,
                                         const int32_t size_f, // size of last dimension of freqs.
                                         const int32_t stride_x_s,
                                         const int32_t stride_x_b,
                                         const int32_t stride_x_h,
                                         const int32_t stride_x_d,
                                         const int32_t stride_y_s,
                                         const int32_t stride_y_b,
                                         const int32_t stride_y_h,
                                         const int32_t stride_y_d)
{
    const uint64_t sid      = blockIdx.x;
    const uint64_t bid      = blockIdx.y;
    const uint64_t offset_x = sid * stride_x_s + bid * stride_x_b;
    const uint64_t offset_y = sid * stride_y_s + bid * stride_y_b;
    const uint64_t offset_f = sid * size_f;

    Op::template apply_2c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDXEq1,
                          StrideDYEq1,
                          StrideDXEq1,
                          StrideDYEq1>(p_inout_x + offset_x,
                                       p_inout_y + offset_y,
                                       p_inout_x + offset_x,
                                       p_inout_y + offset_y,
                                       p_cos + offset_f,
                                       p_sin + offset_f,
                                       size_h_x,
                                       size_h_y,
                                       size_d,
                                       size_f,
                                       stride_x_h,
                                       stride_x_d,
                                       stride_y_h,
                                       stride_y_d,
                                       stride_x_h,
                                       stride_x_d,
                                       stride_y_h,
                                       stride_y_d);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_cached_indirect(scalar_t* __restrict__ p_output,
                                          const scalar_t* __restrict__ p_input,
                                          const scalar_f_t* __restrict__ p_cos,
                                          const scalar_f_t* __restrict__ p_sin,
                                          const int64_t* __restrict__ p_indirect_buffer,
                                          const int32_t max_position,
                                          const int32_t size_h,
                                          const int32_t size_d,
                                          const int32_t size_f, // size of last dimension of freqs.
                                          const int32_t stride_i_s,
                                          const int32_t stride_i_b,
                                          const int32_t stride_i_h,
                                          const int32_t stride_i_d,
                                          const int32_t stride_o_s,
                                          const int32_t stride_o_b,
                                          const int32_t stride_o_h,
                                          const int32_t stride_o_d)
{
    const uint64_t sid    = blockIdx.x;
    const uint64_t bid    = blockIdx.y;
    const uint64_t ib_idx = sid * gridDim.y + bid;
    const int64_t pos     = p_indirect_buffer[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const uint64_t offset_i = sid * stride_i_s + bid * stride_i_b;
        const uint64_t offset_o = sid * stride_o_s + bid * stride_o_b;
        const int64_t offset_f  = pos * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              false,
                              StrideDOutEq1,
                              StrideDInEq1>(p_output + offset_o,
                                            p_input + offset_i,
                                            p_cos + offset_f,
                                            p_sin + offset_f,
                                            size_h,
                                            size_d,
                                            size_f,
                                            stride_i_h,
                                            stride_i_d,
                                            stride_o_h,
                                            stride_o_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutXEq1,
          bool StrideDOutYEq1,
          bool StrideDInXEq1,
          bool StrideDInYEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_cached_indirect(scalar_t* __restrict__ p_output_x,
                                          scalar_t* __restrict__ p_output_y,
                                          const scalar_t* __restrict__ p_input_x,
                                          const scalar_t* __restrict__ p_input_y,
                                          const scalar_f_t* __restrict__ p_cos,
                                          const scalar_f_t* __restrict__ p_sin,
                                          const int64_t* __restrict__ p_indirect_buffer,
                                          const int32_t max_position,
                                          const int32_t size_h_x,
                                          const int32_t size_h_y,
                                          const int32_t size_d,
                                          const int32_t size_f, // size of last dimension of freqs.
                                          const int32_t stride_ix_s,
                                          const int32_t stride_ix_b,
                                          const int32_t stride_ix_h,
                                          const int32_t stride_ix_d,
                                          const int32_t stride_iy_s,
                                          const int32_t stride_iy_b,
                                          const int32_t stride_iy_h,
                                          const int32_t stride_iy_d,
                                          const int32_t stride_ox_s,
                                          const int32_t stride_ox_b,
                                          const int32_t stride_ox_h,
                                          const int32_t stride_ox_d,
                                          const int32_t stride_oy_s,
                                          const int32_t stride_oy_b,
                                          const int32_t stride_oy_h,
                                          const int32_t stride_oy_d)
{
    const uint64_t sid    = blockIdx.x;
    const uint64_t bid    = blockIdx.y;
    const uint64_t ib_idx = sid * gridDim.y + bid;
    const int64_t pos     = p_indirect_buffer[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const uint64_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
        const uint64_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
        const uint64_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
        const uint64_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
        const int64_t offset_f   = pos * size_f;

        Op::template apply_2c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              false,
                              StrideDOutXEq1,
                              StrideDOutYEq1,
                              StrideDInXEq1,
                              StrideDInYEq1>(p_output_x + offset_ox,
                                             p_output_y + offset_oy,
                                             p_input_x + offset_ix,
                                             p_input_y + offset_iy,
                                             p_cos + offset_f,
                                             p_sin + offset_f,
                                             size_h_x,
                                             size_h_y,
                                             size_d,
                                             size_f,
                                             stride_ix_h,
                                             stride_ix_d,
                                             stride_iy_h,
                                             stride_iy_d,
                                             stride_ox_h,
                                             stride_ox_d,
                                             stride_oy_h,
                                             stride_oy_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__ void kn_entry_1c_sbhd_cached_indirect_inplace(
    scalar_t* __restrict__ p_inout,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__ p_indirect_buffer,
    const int32_t max_position,
    const int32_t size_h,
    const int32_t size_d,
    const int32_t size_f, // size of last dimension of freqs.
    const int32_t stride_s,
    const int32_t stride_b,
    const int32_t stride_h,
    const int32_t stride_d)
{
    const uint64_t sid    = blockIdx.x;
    const uint64_t bid    = blockIdx.y;
    const uint64_t ib_idx = sid * gridDim.y + bid;
    const int64_t pos     = p_indirect_buffer[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const uint64_t offset  = sid * stride_s + bid * stride_b;
        const int64_t offset_f = pos * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              true,
                              StrideDEq1,
                              StrideDEq1>(p_inout + offset,
                                          p_inout + offset,
                                          p_cos + offset_f,
                                          p_sin + offset_f,
                                          size_h,
                                          size_d,
                                          size_f,
                                          stride_h,
                                          stride_d,
                                          stride_h,
                                          stride_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDXEq1,
          bool StrideDYEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__ void kn_entry_2c_sbhd_cached_indirect_inplace(
    scalar_t* __restrict__ p_inout_x,
    scalar_t* __restrict__ p_inout_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__ p_indirect_buffer,
    const int32_t max_position,
    const int32_t size_h_x,
    const int32_t size_h_y,
    const int32_t size_d,
    const int32_t size_f, // size of last dimension of freqs.
    const int32_t stride_x_s,
    const int32_t stride_x_b,
    const int32_t stride_x_h,
    const int32_t stride_x_d,
    const int32_t stride_y_s,
    const int32_t stride_y_b,
    const int32_t stride_y_h,
    const int32_t stride_y_d)
{
    const uint64_t sid    = blockIdx.x;
    const uint64_t bid    = blockIdx.y;
    const uint64_t ib_idx = sid * gridDim.y + bid;
    const int64_t pos     = p_indirect_buffer[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const uint64_t offset_x = sid * stride_x_s + bid * stride_x_b;
        const uint64_t offset_y = sid * stride_y_s + bid * stride_y_b;
        const int64_t offset_f  = pos * size_f;

        Op::template apply_2c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              true,
                              StrideDXEq1,
                              StrideDYEq1,
                              StrideDXEq1,
                              StrideDYEq1>(p_inout_x + offset_x,
                                           p_inout_y + offset_y,
                                           p_inout_x + offset_x,
                                           p_inout_y + offset_y,
                                           p_cos + offset_f,
                                           p_sin + offset_f,
                                           size_h_x,
                                           size_h_y,
                                           size_d,
                                           size_f,
                                           stride_x_h,
                                           stride_x_d,
                                           stride_y_h,
                                           stride_y_d,
                                           stride_x_h,
                                           stride_x_d,
                                           stride_y_h,
                                           stride_y_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_sbhd_cached_indirect2(scalar_t* __restrict__ p_output,
                                           const scalar_t* __restrict__ p_input,
                                           const scalar_f_t* __restrict__ p_cos,
                                           const scalar_f_t* __restrict__ p_sin,
                                           const int64_t* __restrict__ p_indirect_buffer_0,
                                           const int64_t* __restrict__ p_indirect_buffer_1,
                                           const int32_t max_position,
                                           const int32_t size_h,
                                           const int32_t size_d,
                                           const int32_t size_f, // size of last dimension of freqs.
                                           const int32_t stride_i_s,
                                           const int32_t stride_i_b,
                                           const int32_t stride_i_h,
                                           const int32_t stride_i_d,
                                           const int32_t stride_o_s,
                                           const int32_t stride_o_b,
                                           const int32_t stride_o_h,
                                           const int32_t stride_o_d)
{
    const uint64_t sid    = blockIdx.x;
    const uint64_t bid    = blockIdx.y;
    const uint64_t ib_idx = sid * gridDim.y + bid;
    const int64_t pos     = p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const uint64_t offset_i = sid * stride_i_s + bid * stride_i_b;
        const uint64_t offset_o = sid * stride_o_s + bid * stride_o_b;
        const int64_t offset_f  = pos * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              false,
                              StrideDOutEq1,
                              StrideDInEq1>(p_output + offset_o,
                                            p_input + offset_i,
                                            p_cos + offset_f,
                                            p_sin + offset_f,
                                            size_h,
                                            size_d,
                                            size_f,
                                            stride_i_h,
                                            stride_i_d,
                                            stride_o_h,
                                            stride_o_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutXEq1,
          bool StrideDOutYEq1,
          bool StrideDInXEq1,
          bool StrideDInYEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_2c_sbhd_cached_indirect2(scalar_t* __restrict__ p_output_x,
                                           scalar_t* __restrict__ p_output_y,
                                           const scalar_t* __restrict__ p_input_x,
                                           const scalar_t* __restrict__ p_input_y,
                                           const scalar_f_t* __restrict__ p_cos,
                                           const scalar_f_t* __restrict__ p_sin,
                                           const int64_t* __restrict__ p_indirect_buffer_0,
                                           const int64_t* __restrict__ p_indirect_buffer_1,
                                           const int32_t max_position,
                                           const int32_t size_h_x,
                                           const int32_t size_h_y,
                                           const int32_t size_d,
                                           const int32_t size_f, // size of last dimension of freqs.
                                           const int32_t stride_ix_s,
                                           const int32_t stride_ix_b,
                                           const int32_t stride_ix_h,
                                           const int32_t stride_ix_d,
                                           const int32_t stride_iy_s,
                                           const int32_t stride_iy_b,
                                           const int32_t stride_iy_h,
                                           const int32_t stride_iy_d,
                                           const int32_t stride_ox_s,
                                           const int32_t stride_ox_b,
                                           const int32_t stride_ox_h,
                                           const int32_t stride_ox_d,
                                           const int32_t stride_oy_s,
                                           const int32_t stride_oy_b,
                                           const int32_t stride_oy_h,
                                           const int32_t stride_oy_d)
{
    const uint64_t sid    = blockIdx.x;
    const uint64_t bid    = blockIdx.y;
    const uint64_t ib_idx = sid * gridDim.y + bid;
    const int64_t pos     = p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const uint64_t offset_ix = sid * stride_ix_s + bid * stride_ix_b;
        const uint64_t offset_iy = sid * stride_iy_s + bid * stride_iy_b;
        const uint64_t offset_ox = sid * stride_ox_s + bid * stride_ox_b;
        const uint64_t offset_oy = sid * stride_oy_s + bid * stride_oy_b;
        const int64_t offset_f   = pos * size_f;

        Op::template apply_2c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              false,
                              StrideDOutXEq1,
                              StrideDOutYEq1,
                              StrideDInXEq1,
                              StrideDInYEq1>(p_output_x + offset_ox,
                                             p_output_y + offset_oy,
                                             p_input_x + offset_ix,
                                             p_input_y + offset_iy,
                                             p_cos + offset_f,
                                             p_sin + offset_f,
                                             size_h_x,
                                             size_h_y,
                                             size_d,
                                             size_f,
                                             stride_ix_h,
                                             stride_ix_d,
                                             stride_iy_h,
                                             stride_iy_d,
                                             stride_ox_h,
                                             stride_ox_d,
                                             stride_oy_h,
                                             stride_oy_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__ void kn_entry_1c_sbhd_cached_indirect2_inplace(
    scalar_t* __restrict__ p_inout,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__ p_indirect_buffer_0,
    const int64_t* __restrict__ p_indirect_buffer_1,
    const int32_t max_position,
    const int32_t size_h,
    const int32_t size_d,
    const int32_t size_f, // size of last dimension of freqs.
    const int32_t stride_s,
    const int32_t stride_b,
    const int32_t stride_h,
    const int32_t stride_d)
{
    const uint64_t sid    = blockIdx.x;
    const uint64_t bid    = blockIdx.y;
    const uint64_t ib_idx = sid * gridDim.y + bid;
    const int64_t pos     = p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const uint64_t offset  = sid * stride_s + bid * stride_b;
        const int64_t offset_f = pos * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              true,
                              StrideDEq1,
                              StrideDEq1>(p_inout + offset,
                                          p_inout + offset,
                                          p_cos + offset_f,
                                          p_sin + offset_f,
                                          size_h,
                                          size_d,
                                          size_f,
                                          stride_h,
                                          stride_d,
                                          stride_h,
                                          stride_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDXEq1,
          bool StrideDYEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__ void kn_entry_2c_sbhd_cached_indirect2_inplace(
    scalar_t* __restrict__ p_inout_x,
    scalar_t* __restrict__ p_inout_y,
    const scalar_f_t* __restrict__ p_cos,
    const scalar_f_t* __restrict__ p_sin,
    const int64_t* __restrict__ p_indirect_buffer_0,
    const int64_t* __restrict__ p_indirect_buffer_1,
    const int32_t max_position,
    const int32_t size_h_x,
    const int32_t size_h_y,
    const int32_t size_d,
    const int32_t size_f, // size of last dimension of freqs.
    const int32_t stride_x_s,
    const int32_t stride_x_b,
    const int32_t stride_x_h,
    const int32_t stride_x_d,
    const int32_t stride_y_s,
    const int32_t stride_y_b,
    const int32_t stride_y_h,
    const int32_t stride_y_d)
{
    const uint64_t sid    = blockIdx.x;
    const uint64_t bid    = blockIdx.y;
    const uint64_t ib_idx = sid * gridDim.y + bid;
    const int64_t pos     = p_indirect_buffer_0[ib_idx] + p_indirect_buffer_1[ib_idx];

    if((pos >= 0) && (pos < max_position))
    {
        const uint64_t offset_x = sid * stride_x_s + bid * stride_x_b;
        const uint64_t offset_y = sid * stride_y_s + bid * stride_y_b;
        const int64_t offset_f  = pos * size_f;

        Op::template apply_2c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              true,
                              StrideDXEq1,
                              StrideDYEq1,
                              StrideDXEq1,
                              StrideDYEq1>(p_inout_x + offset_x,
                                           p_inout_y + offset_y,
                                           p_inout_x + offset_x,
                                           p_inout_y + offset_y,
                                           p_cos + offset_f,
                                           p_sin + offset_f,
                                           size_h_x,
                                           size_h_y,
                                           size_d,
                                           size_f,
                                           stride_x_h,
                                           stride_x_d,
                                           stride_y_h,
                                           stride_y_d,
                                           stride_x_h,
                                           stride_x_d,
                                           stride_y_h,
                                           stride_y_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_thd_uncached(scalar_t* __restrict__ p_output,
                                  const scalar_t* __restrict__ p_input,
                                  const int32_t* __restrict__ p_cu_seqlens,
                                  const scalar_f_t* __restrict__ p_freqs,
                                  const int32_t size_h,
                                  const int32_t size_d,
                                  const int32_t size_f, // size of last dimension of freqs.
                                  const int32_t stride_i_t,
                                  const int32_t stride_i_h,
                                  const int32_t stride_i_d,
                                  const int32_t stride_o_t,
                                  const int32_t stride_o_h,
                                  const int32_t stride_o_d)
{
    const uint64_t sid = blockIdx.x;
    const uint64_t bid = blockIdx.y;
    const uint64_t tid = sid + p_cu_seqlens[bid];

    if(tid < p_cu_seqlens[bid + 1])
    {
        const int32_t offset_i = tid * stride_i_t;
        const int32_t offset_o = tid * stride_o_t;
        const int32_t offset_f = sid * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              false,
                              StrideDOutEq1,
                              StrideDInEq1>(p_output + offset_o,
                                            p_input + offset_i,
                                            p_freqs + offset_f,
                                            size_h,
                                            size_d,
                                            size_f,
                                            stride_i_h,
                                            stride_i_d,
                                            stride_o_h,
                                            stride_o_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_thd_uncached_inplace(scalar_t* __restrict__ p_inout,
                                          const int32_t* __restrict__ p_cu_seqlens,
                                          const scalar_f_t* __restrict__ p_freqs,
                                          const int32_t size_h,
                                          const int32_t size_d,
                                          const int32_t size_f, // size of last dimension of freqs.
                                          const int32_t stride_t,
                                          const int32_t stride_h,
                                          const int32_t stride_d)
{
    const uint64_t sid = blockIdx.x;
    const uint64_t bid = blockIdx.y;
    const uint64_t tid = sid + p_cu_seqlens[bid];

    if(tid < p_cu_seqlens[bid + 1])
    {
        const int32_t offset   = tid * stride_t;
        const int32_t offset_f = sid * size_f;

        Op::template apply_1c<RotateStyle,
                              ReuseFreqsFrontPart,
                              NopeFirst,
                              true,
                              StrideDEq1,
                              StrideDEq1>(p_inout + offset,
                                          p_inout + offset,
                                          p_freqs + offset_f,
                                          size_h,
                                          size_d,
                                          size_f,
                                          stride_h,
                                          stride_d,
                                          stride_h,
                                          stride_d);
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDOutEq1,
          bool StrideDInEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_2d_cached(scalar_t* __restrict__ p_output,
                               const scalar_t* __restrict__ p_input,
                               const scalar_f_t* __restrict__ p_cos_h,
                               const scalar_f_t* __restrict__ p_sin_h,
                               const scalar_f_t* __restrict__ p_cos_w,
                               const scalar_f_t* __restrict__ p_sin_w,
                               const int32_t img_width,
                               const int32_t size_h,
                               const int32_t size_d,
                               const int32_t stride_i_b,
                               const int32_t stride_i_s,
                               const int32_t stride_i_h,
                               const int32_t stride_i_d,
                               const int32_t stride_o_b,
                               const int32_t stride_o_s,
                               const int32_t stride_o_h,
                               const int32_t stride_o_d)
{
    const uint64_t Hid         = blockIdx.x;
    const uint64_t Wid         = blockIdx.y;
    const uint64_t sid         = Hid * img_width + Wid;
    const uint64_t bid         = blockIdx.z;
    const uint64_t size_half_d = size_d >> 1;

    const int offset_h_i = bid * stride_i_b + sid * stride_i_s;
    const int offset_h_o = bid * stride_o_b + sid * stride_o_s;
    const int offset_h_f = Hid * size_half_d;
    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutEq1,
                          StrideDInEq1>(p_output + offset_h_o,
                                        p_input + offset_h_i,
                                        p_cos_h + offset_h_f,
                                        p_sin_h + offset_h_f,
                                        size_h,
                                        size_half_d,
                                        size_half_d,
                                        stride_i_h,
                                        stride_i_d,
                                        stride_o_h,
                                        stride_o_d);

    const int offset_w_i = offset_h_i + size_half_d * stride_i_d;
    const int offset_w_o = offset_h_o + size_half_d * stride_o_d;
    const int offset_w_f = Wid * size_half_d;
    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          false,
                          StrideDOutEq1,
                          StrideDInEq1>(p_output + offset_w_o,
                                        p_input + offset_w_i,
                                        p_cos_w + offset_w_f,
                                        p_sin_w + offset_w_f,
                                        size_h,
                                        size_half_d,
                                        size_half_d,
                                        stride_i_h,
                                        stride_i_d,
                                        stride_o_h,
                                        stride_o_d);
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          bool StrideDEq1,
          typename scalar_t,
          typename scalar_f_t>
__launch_bounds__(256, 8) __global__
    void kn_entry_1c_2d_cached_inplace(scalar_t* __restrict__ p_inout,
                                       const scalar_f_t* __restrict__ p_cos_h,
                                       const scalar_f_t* __restrict__ p_sin_h,
                                       const scalar_f_t* __restrict__ p_cos_w,
                                       const scalar_f_t* __restrict__ p_sin_w,
                                       const int32_t img_width,
                                       const int32_t size_h,
                                       const int32_t size_d,
                                       const int32_t stride_b,
                                       const int32_t stride_s,
                                       const int32_t stride_h,
                                       const int32_t stride_d)
{
    const uint64_t Hid         = blockIdx.x;
    const uint64_t Wid         = blockIdx.y;
    const uint64_t sid         = Hid * img_width + Wid;
    const uint64_t bid         = blockIdx.z;
    const uint64_t size_half_d = size_d >> 1;

    const int offset_h   = bid * stride_b + sid * stride_s;
    const int offset_h_f = Hid * size_half_d;
    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDEq1,
                          StrideDEq1>(p_inout + offset_h,
                                      p_inout + offset_h,
                                      p_cos_h + offset_h_f,
                                      p_sin_h + offset_h_f,
                                      size_h,
                                      size_half_d,
                                      size_half_d,
                                      stride_h,
                                      stride_d,
                                      stride_h,
                                      stride_d);

    const int offset_w   = offset_h + size_half_d * stride_d;
    const int offset_w_f = Wid * size_half_d;
    Op::template apply_1c<RotateStyle,
                          ReuseFreqsFrontPart,
                          NopeFirst,
                          true,
                          StrideDEq1,
                          StrideDEq1>(p_inout + offset_w,
                                      p_inout + offset_w,
                                      p_cos_w + offset_w_f,
                                      p_sin_w + offset_w_f,
                                      size_h,
                                      size_half_d,
                                      size_half_d,
                                      stride_h,
                                      stride_d,
                                      stride_h,
                                      stride_d);
}

// =====================================================================================================================
// Dispatches
//

#define LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(ROTATE_STYLE, STRIDE_0, ...) \
    if constexpr((ROTATE_STYLE) != ROTATE_STYLE_GPTJ)                       \
    {                                                                       \
        constexpr bool Stride0Eq1 = false;                                  \
        __VA_ARGS__;                                                        \
    }                                                                       \
    else if((STRIDE_0) == 1)                                                \
    {                                                                       \
        constexpr bool Stride0Eq1 = true;                                   \
        __VA_ARGS__;                                                        \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        constexpr bool Stride0Eq1 = false;                                  \
        __VA_ARGS__;                                                        \
    }

#define LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(ROTATE_STYLE, STRIDE_0, STRIDE_1, ...) \
    if constexpr((ROTATE_STYLE) != ROTATE_STYLE_GPTJ)                                 \
    {                                                                                 \
        constexpr bool Stride0Eq1 = false;                                            \
        constexpr bool Stride1Eq1 = false;                                            \
        __VA_ARGS__;                                                                  \
    }                                                                                 \
    else if((STRIDE_0) == 1)                                                          \
    {                                                                                 \
        constexpr bool Stride0Eq1 = true;                                             \
        if((STRIDE_1) == 1)                                                           \
        {                                                                             \
            constexpr bool Stride1Eq1 = true;                                         \
            __VA_ARGS__;                                                              \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            constexpr bool Stride1Eq1 = false;                                        \
            __VA_ARGS__;                                                              \
        }                                                                             \
    }                                                                                 \
    else                                                                              \
    {                                                                                 \
        constexpr bool Stride0Eq1 = false;                                            \
        if((STRIDE_1) == 1)                                                           \
        {                                                                             \
            constexpr bool Stride1Eq1 = true;                                         \
            __VA_ARGS__;                                                              \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            constexpr bool Stride1Eq1 = false;                                        \
            __VA_ARGS__;                                                              \
        }                                                                             \
    }

#define LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(                \
    ROTATE_STYLE, STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3, ...) \
    if constexpr((ROTATE_STYLE) != ROTATE_STYLE_GPTJ)          \
    {                                                          \
        constexpr bool Stride0Eq1 = false;                     \
        constexpr bool Stride1Eq1 = false;                     \
        constexpr bool Stride2Eq1 = false;                     \
        constexpr bool Stride3Eq1 = false;                     \
        __VA_ARGS__;                                           \
    }                                                          \
    else if((STRIDE_0) == 1)                                   \
    {                                                          \
        constexpr bool Stride0Eq1 = true;                      \
        if((STRIDE_1) == 1)                                    \
        {                                                      \
            constexpr bool Stride1Eq1 = true;                  \
            if((STRIDE_2) == 1)                                \
            {                                                  \
                constexpr bool Stride2Eq1 = true;              \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
            else                                               \
            {                                                  \
                constexpr bool Stride2Eq1 = false;             \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
        }                                                      \
        else                                                   \
        {                                                      \
            constexpr bool Stride1Eq1 = false;                 \
            if((STRIDE_2) == 1)                                \
            {                                                  \
                constexpr bool Stride2Eq1 = true;              \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
            else                                               \
            {                                                  \
                constexpr bool Stride2Eq1 = false;             \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
        }                                                      \
    }                                                          \
    else                                                       \
    {                                                          \
        constexpr bool Stride0Eq1 = false;                     \
        if((STRIDE_1) == 1)                                    \
        {                                                      \
            constexpr bool Stride1Eq1 = true;                  \
            if((STRIDE_2) == 1)                                \
            {                                                  \
                constexpr bool Stride2Eq1 = true;              \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
            else                                               \
            {                                                  \
                constexpr bool Stride2Eq1 = false;             \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
        }                                                      \
        else                                                   \
        {                                                      \
            constexpr bool Stride1Eq1 = false;                 \
            if((STRIDE_2) == 1)                                \
            {                                                  \
                constexpr bool Stride2Eq1 = true;              \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
            else                                               \
            {                                                  \
                constexpr bool Stride2Eq1 = false;             \
                if((STRIDE_3) == 1)                            \
                {                                              \
                    constexpr bool Stride3Eq1 = true;          \
                    __VA_ARGS__;                               \
                }                                              \
                else                                           \
                {                                              \
                    constexpr bool Stride3Eq1 = false;         \
                    __VA_ARGS__;                               \
                }                                              \
            }                                                  \
        }                                                      \
    }

template <bool ReuseFreqsFrontPart, bool Is2D>
std::tuple<dim3, dim3> get_grid_config(const int32_t size_s_h,
                                       const int32_t size_s_w,
                                       const int32_t size_b,
                                       const int32_t size_f)
{
    constexpr int32_t num_warps   = 4;
    constexpr int32_t num_threads = num_warps * ck_tile::get_warp_size();

    const int32_t size_r              = ReuseFreqsFrontPart ? (size_f << 1) : size_f;
    const int32_t size_half_r         = size_r >> 1;
    const int32_t aligned_size_half_r = ck_tile::next_power_of_two(size_half_r);

    const int32_t block_dim_x = std::min(aligned_size_half_r, ck_tile::get_warp_size());
    const int32_t block_dim_y = std::max(num_threads / block_dim_x, 1);

    if constexpr(Is2D)
    {
        return {dim3(size_s_h, size_s_w, size_b), dim3(block_dim_x, block_dim_y)};
    }
    else
    {
        return {dim3(size_s_h * size_s_w, size_b), dim3(block_dim_x, block_dim_y)};
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_sbhd_uncached(scalar_t* __restrict__ p_output,
                               const scalar_t* __restrict__ p_input,
                               const scalar_f_t* __restrict__ p_freqs,
                               const int32_t size_s,
                               const int32_t size_b,
                               const int32_t size_h,
                               const int32_t size_d,
                               const int32_t size_f, // size of last dimension of freqs.
                               const int32_t stride_i_s,
                               const int32_t stride_i_b,
                               const int32_t stride_i_h,
                               const int32_t stride_i_d,
                               const int32_t stride_o_s,
                               const int32_t stride_o_b,
                               const int32_t stride_o_h,
                               const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] = get_grid_config<ReuseFreqsFrontPart, false>(size_s, 1, size_b, size_f);

    if(p_output == p_input)
    {
        assert(stride_i_s == stride_o_s);
        assert(stride_i_b == stride_o_b);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
            RotateStyle,
            stride_i_d,
            kn_entry_1c_sbhd_uncached_inplace<Op,
                                              RotateStyle,
                                              ReuseFreqsFrontPart,
                                              NopeFirst,
                                              Stride0Eq1><<<grid, block, 0, stream>>>(p_output,
                                                                                      p_freqs,
                                                                                      size_h,
                                                                                      size_d,
                                                                                      size_f,
                                                                                      stride_i_s,
                                                                                      stride_i_b,
                                                                                      stride_i_h,
                                                                                      stride_i_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(RotateStyle,
                                               stride_o_d,
                                               stride_i_d,
                                               kn_entry_1c_sbhd_uncached<Op,
                                                                         RotateStyle,
                                                                         ReuseFreqsFrontPart,
                                                                         NopeFirst,
                                                                         Stride0Eq1,
                                                                         Stride1Eq1>
                                               <<<grid, block, 0, stream>>>(p_output,
                                                                            p_input,
                                                                            p_freqs,
                                                                            size_h,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_i_s,
                                                                            stride_i_b,
                                                                            stride_i_h,
                                                                            stride_i_d,
                                                                            stride_o_s,
                                                                            stride_o_b,
                                                                            stride_o_h,
                                                                            stride_o_d););
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_2c_sbhd_uncached(scalar_t* __restrict__ p_output_x,
                               scalar_t* __restrict__ p_output_y,
                               const scalar_t* __restrict__ p_input_x,
                               const scalar_t* __restrict__ p_input_y,
                               const scalar_f_t* __restrict__ p_freqs,
                               const int32_t size_s,
                               const int32_t size_b,
                               const int32_t size_h_x,
                               const int32_t size_h_y,
                               const int32_t size_d,
                               const int32_t size_f, // size of last dimension of freqs.
                               const int32_t stride_ix_s,
                               const int32_t stride_ix_b,
                               const int32_t stride_ix_h,
                               const int32_t stride_ix_d,
                               const int32_t stride_iy_s,
                               const int32_t stride_iy_b,
                               const int32_t stride_iy_h,
                               const int32_t stride_iy_d,
                               const int32_t stride_ox_s,
                               const int32_t stride_ox_b,
                               const int32_t stride_ox_h,
                               const int32_t stride_ox_d,
                               const int32_t stride_oy_s,
                               const int32_t stride_oy_b,
                               const int32_t stride_oy_h,
                               const int32_t stride_oy_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] = get_grid_config<ReuseFreqsFrontPart, false>(size_s, 1, size_b, size_f);

    if((p_output_x == p_input_x) && (p_output_y == p_input_y))
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
            kn_entry_2c_sbhd_uncached_inplace<Op,
                                              RotateStyle,
                                              ReuseFreqsFrontPart,
                                              NopeFirst,
                                              Stride0Eq1,
                                              Stride1Eq1>
            <<<grid, block, 0, stream>>>(p_output_x,
                                         p_output_y,
                                         p_freqs,
                                         size_h_x,
                                         size_h_y,
                                         size_d,
                                         size_f,
                                         stride_ix_s,
                                         stride_ix_b,
                                         stride_ix_h,
                                         stride_ix_d,
                                         stride_iy_s,
                                         stride_iy_b,
                                         stride_iy_h,
                                         stride_iy_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(RotateStyle,
                                               stride_ox_d,
                                               stride_oy_d,
                                               stride_ix_d,
                                               stride_iy_d,
                                               kn_entry_2c_sbhd_uncached<Op,
                                                                         RotateStyle,
                                                                         ReuseFreqsFrontPart,
                                                                         NopeFirst,
                                                                         Stride0Eq1,
                                                                         Stride1Eq1,
                                                                         Stride2Eq1,
                                                                         Stride3Eq1>
                                               <<<grid, block, 0, stream>>>(p_output_x,
                                                                            p_output_y,
                                                                            p_input_x,
                                                                            p_input_y,
                                                                            p_freqs,
                                                                            size_h_x,
                                                                            size_h_y,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_ix_s,
                                                                            stride_ix_b,
                                                                            stride_ix_h,
                                                                            stride_ix_d,
                                                                            stride_iy_s,
                                                                            stride_iy_b,
                                                                            stride_iy_h,
                                                                            stride_iy_d,
                                                                            stride_ox_s,
                                                                            stride_ox_b,
                                                                            stride_ox_h,
                                                                            stride_ox_d,
                                                                            stride_oy_s,
                                                                            stride_oy_b,
                                                                            stride_oy_h,
                                                                            stride_oy_d););
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_sbhd_cached(scalar_t* __restrict__ p_output,
                             const scalar_t* __restrict__ p_input,
                             const scalar_f_t* __restrict__ p_cos,
                             const scalar_f_t* __restrict__ p_sin,
                             const int32_t size_s,
                             const int32_t size_b,
                             const int32_t size_h,
                             const int32_t size_d,
                             const int32_t size_f, // size of last dimension of freqs.
                             const int32_t stride_i_s,
                             const int32_t stride_i_b,
                             const int32_t stride_i_h,
                             const int32_t stride_i_d,
                             const int32_t stride_o_s,
                             const int32_t stride_o_b,
                             const int32_t stride_o_h,
                             const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] = get_grid_config<ReuseFreqsFrontPart, false>(size_s, 1, size_b, size_f);

    if(p_output == p_input)
    {
        assert(stride_i_s == stride_o_s);
        assert(stride_i_b == stride_o_b);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(RotateStyle,
                                               stride_i_d,
                                               kn_entry_1c_sbhd_cached_inplace<Op,
                                                                               RotateStyle,
                                                                               ReuseFreqsFrontPart,
                                                                               NopeFirst,
                                                                               Stride0Eq1>
                                               <<<grid, block, 0, stream>>>(p_output,
                                                                            p_cos,
                                                                            p_sin,
                                                                            size_h,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_i_s,
                                                                            stride_i_b,
                                                                            stride_i_h,
                                                                            stride_i_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(RotateStyle,
                                               stride_o_d,
                                               stride_i_d,
                                               kn_entry_1c_sbhd_cached<Op,
                                                                       RotateStyle,
                                                                       ReuseFreqsFrontPart,
                                                                       NopeFirst,
                                                                       Stride0Eq1,
                                                                       Stride1Eq1>
                                               <<<grid, block, 0, stream>>>(p_output,
                                                                            p_input,
                                                                            p_cos,
                                                                            p_sin,
                                                                            size_h,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_i_s,
                                                                            stride_i_b,
                                                                            stride_i_h,
                                                                            stride_i_d,
                                                                            stride_o_s,
                                                                            stride_o_b,
                                                                            stride_o_h,
                                                                            stride_o_d););
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_2c_sbhd_cached(scalar_t* __restrict__ p_output_x,
                             scalar_t* __restrict__ p_output_y,
                             const scalar_t* __restrict__ p_input_x,
                             const scalar_t* __restrict__ p_input_y,
                             const scalar_f_t* __restrict__ p_cos,
                             const scalar_f_t* __restrict__ p_sin,
                             const int32_t size_s,
                             const int32_t size_b,
                             const int32_t size_h_x,
                             const int32_t size_h_y,
                             const int32_t size_d,
                             const int32_t size_f, // size of last dimension of freqs.
                             const int32_t stride_ix_s,
                             const int32_t stride_ix_b,
                             const int32_t stride_ix_h,
                             const int32_t stride_ix_d,
                             const int32_t stride_iy_s,
                             const int32_t stride_iy_b,
                             const int32_t stride_iy_h,
                             const int32_t stride_iy_d,
                             const int32_t stride_ox_s,
                             const int32_t stride_ox_b,
                             const int32_t stride_ox_h,
                             const int32_t stride_ox_d,
                             const int32_t stride_oy_s,
                             const int32_t stride_oy_b,
                             const int32_t stride_oy_h,
                             const int32_t stride_oy_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] = get_grid_config<ReuseFreqsFrontPart, false>(size_s, 1, size_b, size_f);

    if((p_output_x == p_input_x) && (p_output_y == p_input_y))
    {
        assert(stride_ix_s == stride_ox_s);
        assert(stride_ix_b == stride_ox_b);
        assert(stride_ix_h == stride_ox_h);
        assert(stride_ix_d == stride_ox_d);
        assert(stride_iy_s == stride_oy_s);
        assert(stride_iy_b == stride_oy_b);
        assert(stride_iy_h == stride_oy_h);
        assert(stride_iy_d == stride_oy_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(RotateStyle,
                                               stride_ix_d,
                                               stride_iy_d,
                                               kn_entry_2c_sbhd_cached_inplace<Op,
                                                                               RotateStyle,
                                                                               ReuseFreqsFrontPart,
                                                                               NopeFirst,
                                                                               Stride0Eq1,
                                                                               Stride1Eq1>
                                               <<<grid, block, 0, stream>>>(p_output_x,
                                                                            p_output_y,
                                                                            p_cos,
                                                                            p_sin,
                                                                            size_h_x,
                                                                            size_h_y,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_ix_s,
                                                                            stride_ix_b,
                                                                            stride_ix_h,
                                                                            stride_ix_d,
                                                                            stride_iy_s,
                                                                            stride_iy_b,
                                                                            stride_iy_h,
                                                                            stride_iy_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(RotateStyle,
                                               stride_ox_d,
                                               stride_oy_d,
                                               stride_ix_d,
                                               stride_iy_d,
                                               kn_entry_2c_sbhd_cached<Op,
                                                                       RotateStyle,
                                                                       ReuseFreqsFrontPart,
                                                                       NopeFirst,
                                                                       Stride0Eq1,
                                                                       Stride1Eq1,
                                                                       Stride2Eq1,
                                                                       Stride3Eq1>
                                               <<<grid, block, 0, stream>>>(p_output_x,
                                                                            p_output_y,
                                                                            p_input_x,
                                                                            p_input_y,
                                                                            p_cos,
                                                                            p_sin,
                                                                            size_h_x,
                                                                            size_h_y,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_ix_s,
                                                                            stride_ix_b,
                                                                            stride_ix_h,
                                                                            stride_ix_d,
                                                                            stride_iy_s,
                                                                            stride_iy_b,
                                                                            stride_iy_h,
                                                                            stride_iy_d,
                                                                            stride_ox_s,
                                                                            stride_ox_b,
                                                                            stride_ox_h,
                                                                            stride_ox_d,
                                                                            stride_oy_s,
                                                                            stride_oy_b,
                                                                            stride_oy_h,
                                                                            stride_oy_d););
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_sbhd_cached_indirect(scalar_t* __restrict__ p_output,
                                      const scalar_t* __restrict__ p_input,
                                      const scalar_f_t* __restrict__ p_cos,
                                      const scalar_f_t* __restrict__ p_sin,
                                      const int64_t* __restrict__ p_indirect_buffer,
                                      const int32_t max_position,
                                      const int32_t size_s,
                                      const int32_t size_b,
                                      const int32_t size_h,
                                      const int32_t size_d,
                                      const int32_t size_f, // size of last dimension of freqs.
                                      const int32_t stride_i_s,
                                      const int32_t stride_i_b,
                                      const int32_t stride_i_h,
                                      const int32_t stride_i_d,
                                      const int32_t stride_o_s,
                                      const int32_t stride_o_b,
                                      const int32_t stride_o_h,
                                      const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] = get_grid_config<ReuseFreqsFrontPart, false>(size_s, 1, size_b, size_f);

    if(p_output == p_input)
    {
        assert(stride_i_s == stride_o_s);
        assert(stride_i_b == stride_o_b);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
            RotateStyle,
            stride_i_d,
            kn_entry_1c_sbhd_cached_indirect_inplace<Op,
                                                     RotateStyle,
                                                     ReuseFreqsFrontPart,
                                                     NopeFirst,
                                                     Stride0Eq1>
            <<<grid, block, 0, stream>>>(p_output,
                                         p_cos,
                                         p_sin,
                                         p_indirect_buffer,
                                         max_position,
                                         size_h,
                                         size_d,
                                         size_f,
                                         stride_i_s,
                                         stride_i_b,
                                         stride_i_h,
                                         stride_i_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(RotateStyle,
                                               stride_o_d,
                                               stride_i_d,
                                               kn_entry_1c_sbhd_cached_indirect<Op,
                                                                                RotateStyle,
                                                                                ReuseFreqsFrontPart,
                                                                                NopeFirst,
                                                                                Stride0Eq1,
                                                                                Stride1Eq1>
                                               <<<grid, block, 0, stream>>>(p_output,
                                                                            p_input,
                                                                            p_cos,
                                                                            p_sin,
                                                                            p_indirect_buffer,
                                                                            max_position,
                                                                            size_h,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_i_s,
                                                                            stride_i_b,
                                                                            stride_i_h,
                                                                            stride_i_d,
                                                                            stride_o_s,
                                                                            stride_o_b,
                                                                            stride_o_h,
                                                                            stride_o_d););
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_2c_sbhd_cached_indirect(scalar_t* __restrict__ p_output_x,
                                      scalar_t* __restrict__ p_output_y,
                                      const scalar_t* __restrict__ p_input_x,
                                      const scalar_t* __restrict__ p_input_y,
                                      const scalar_f_t* __restrict__ p_cos,
                                      const scalar_f_t* __restrict__ p_sin,
                                      const int64_t* __restrict__ p_indirect_buffer,
                                      const int32_t max_position,
                                      const int32_t size_s,
                                      const int32_t size_b,
                                      const int32_t size_h_x,
                                      const int32_t size_h_y,
                                      const int32_t size_d,
                                      const int32_t size_f, // size of last dimension of freqs.
                                      const int32_t stride_ix_s,
                                      const int32_t stride_ix_b,
                                      const int32_t stride_ix_h,
                                      const int32_t stride_ix_d,
                                      const int32_t stride_iy_s,
                                      const int32_t stride_iy_b,
                                      const int32_t stride_iy_h,
                                      const int32_t stride_iy_d,
                                      const int32_t stride_ox_s,
                                      const int32_t stride_ox_b,
                                      const int32_t stride_ox_h,
                                      const int32_t stride_ox_d,
                                      const int32_t stride_oy_s,
                                      const int32_t stride_oy_b,
                                      const int32_t stride_oy_h,
                                      const int32_t stride_oy_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] = get_grid_config<ReuseFreqsFrontPart, false>(size_s, 1, size_b, size_f);

    if((p_output_x == p_input_x) && (p_output_y == p_input_y))
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
            kn_entry_2c_sbhd_cached_indirect_inplace<Op,
                                                     RotateStyle,
                                                     ReuseFreqsFrontPart,
                                                     NopeFirst,
                                                     Stride0Eq1,
                                                     Stride1Eq1>
            <<<grid, block, 0, stream>>>(p_output_x,
                                         p_output_y,
                                         p_cos,
                                         p_sin,
                                         p_indirect_buffer,
                                         max_position,
                                         size_h_x,
                                         size_h_y,
                                         size_d,
                                         size_f,
                                         stride_ix_s,
                                         stride_ix_b,
                                         stride_ix_h,
                                         stride_ix_d,
                                         stride_iy_s,
                                         stride_iy_b,
                                         stride_iy_h,
                                         stride_iy_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(RotateStyle,
                                               stride_ox_d,
                                               stride_oy_d,
                                               stride_ix_d,
                                               stride_iy_d,
                                               kn_entry_2c_sbhd_cached_indirect<Op,
                                                                                RotateStyle,
                                                                                ReuseFreqsFrontPart,
                                                                                NopeFirst,
                                                                                Stride0Eq1,
                                                                                Stride1Eq1,
                                                                                Stride2Eq1,
                                                                                Stride3Eq1>
                                               <<<grid, block, 0, stream>>>(p_output_x,
                                                                            p_output_y,
                                                                            p_input_x,
                                                                            p_input_y,
                                                                            p_cos,
                                                                            p_sin,
                                                                            p_indirect_buffer,
                                                                            max_position,
                                                                            size_h_x,
                                                                            size_h_y,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_ix_s,
                                                                            stride_ix_b,
                                                                            stride_ix_h,
                                                                            stride_ix_d,
                                                                            stride_iy_s,
                                                                            stride_iy_b,
                                                                            stride_iy_h,
                                                                            stride_iy_d,
                                                                            stride_ox_s,
                                                                            stride_ox_b,
                                                                            stride_ox_h,
                                                                            stride_ox_d,
                                                                            stride_oy_s,
                                                                            stride_oy_b,
                                                                            stride_oy_h,
                                                                            stride_oy_d););
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_sbhd_cached_indirect2(scalar_t* __restrict__ p_output,
                                       const scalar_t* __restrict__ p_input,
                                       const scalar_f_t* __restrict__ p_cos,
                                       const scalar_f_t* __restrict__ p_sin,
                                       const int64_t* __restrict__ p_indirect_buffer_0,
                                       const int64_t* __restrict__ p_indirect_buffer_1,
                                       const int32_t max_position,
                                       const int32_t size_s,
                                       const int32_t size_b,
                                       const int32_t size_h,
                                       const int32_t size_d,
                                       const int32_t size_f, // size of last dimension of freqs.
                                       const int32_t stride_i_s,
                                       const int32_t stride_i_b,
                                       const int32_t stride_i_h,
                                       const int32_t stride_i_d,
                                       const int32_t stride_o_s,
                                       const int32_t stride_o_b,
                                       const int32_t stride_o_h,
                                       const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] = get_grid_config<ReuseFreqsFrontPart, false>(size_s, 1, size_b, size_f);

    if(p_output == p_input)
    {
        assert(stride_i_s == stride_o_s);
        assert(stride_i_b == stride_o_b);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(
            RotateStyle,
            stride_i_d,
            kn_entry_1c_sbhd_cached_indirect2_inplace<Op,
                                                      RotateStyle,
                                                      ReuseFreqsFrontPart,
                                                      NopeFirst,
                                                      Stride0Eq1>
            <<<grid, block, 0, stream>>>(p_output,
                                         p_cos,
                                         p_sin,
                                         p_indirect_buffer_0,
                                         p_indirect_buffer_1,
                                         max_position,
                                         size_h,
                                         size_d,
                                         size_f,
                                         stride_i_s,
                                         stride_i_b,
                                         stride_i_h,
                                         stride_i_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(
            RotateStyle,
            stride_o_d,
            stride_i_d,
            kn_entry_1c_sbhd_cached_indirect2<Op,
                                              RotateStyle,
                                              ReuseFreqsFrontPart,
                                              NopeFirst,
                                              Stride0Eq1,
                                              Stride1Eq1>
            <<<grid, block, 0, stream>>>(p_output,
                                         p_input,
                                         p_cos,
                                         p_sin,
                                         p_indirect_buffer_0,
                                         p_indirect_buffer_1,
                                         max_position,
                                         size_h,
                                         size_d,
                                         size_f,
                                         stride_i_s,
                                         stride_i_b,
                                         stride_i_h,
                                         stride_i_d,
                                         stride_o_s,
                                         stride_o_b,
                                         stride_o_h,
                                         stride_o_d););
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_2c_sbhd_cached_indirect2(scalar_t* __restrict__ p_output_x,
                                       scalar_t* __restrict__ p_output_y,
                                       const scalar_t* __restrict__ p_input_x,
                                       const scalar_t* __restrict__ p_input_y,
                                       const scalar_f_t* __restrict__ p_cos,
                                       const scalar_f_t* __restrict__ p_sin,
                                       const int64_t* __restrict__ p_indirect_buffer_0,
                                       const int64_t* __restrict__ p_indirect_buffer_1,
                                       const int32_t max_position,
                                       const int32_t size_s,
                                       const int32_t size_b,
                                       const int32_t size_h_x,
                                       const int32_t size_h_y,
                                       const int32_t size_d,
                                       const int32_t size_f, // size of last dimension of freqs.
                                       const int32_t stride_ix_s,
                                       const int32_t stride_ix_b,
                                       const int32_t stride_ix_h,
                                       const int32_t stride_ix_d,
                                       const int32_t stride_iy_s,
                                       const int32_t stride_iy_b,
                                       const int32_t stride_iy_h,
                                       const int32_t stride_iy_d,
                                       const int32_t stride_ox_s,
                                       const int32_t stride_ox_b,
                                       const int32_t stride_ox_h,
                                       const int32_t stride_ox_d,
                                       const int32_t stride_oy_s,
                                       const int32_t stride_oy_b,
                                       const int32_t stride_oy_h,
                                       const int32_t stride_oy_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] = get_grid_config<ReuseFreqsFrontPart, false>(size_s, 1, size_b, size_f);

    if((p_output_x == p_input_x) && (p_output_y == p_input_y))
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
            kn_entry_2c_sbhd_cached_indirect2_inplace<Op,
                                                      RotateStyle,
                                                      ReuseFreqsFrontPart,
                                                      NopeFirst,
                                                      Stride0Eq1,
                                                      Stride1Eq1>
            <<<grid, block, 0, stream>>>(p_output_x,
                                         p_output_y,
                                         p_cos,
                                         p_sin,
                                         p_indirect_buffer_0,
                                         p_indirect_buffer_1,
                                         max_position,
                                         size_h_x,
                                         size_h_y,
                                         size_d,
                                         size_f,
                                         stride_ix_s,
                                         stride_ix_b,
                                         stride_ix_h,
                                         stride_ix_d,
                                         stride_iy_s,
                                         stride_iy_b,
                                         stride_iy_h,
                                         stride_iy_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_4_STRIDES(
            RotateStyle,
            stride_ox_d,
            stride_oy_d,
            stride_ix_d,
            stride_iy_d,
            kn_entry_2c_sbhd_cached_indirect2<Op,
                                              RotateStyle,
                                              ReuseFreqsFrontPart,
                                              NopeFirst,
                                              Stride0Eq1,
                                              Stride1Eq1,
                                              Stride2Eq1,
                                              Stride3Eq1>
            <<<grid, block, 0, stream>>>(p_output_x,
                                         p_output_y,
                                         p_input_x,
                                         p_input_y,
                                         p_cos,
                                         p_sin,
                                         p_indirect_buffer_0,
                                         p_indirect_buffer_1,
                                         max_position,
                                         size_h_x,
                                         size_h_y,
                                         size_d,
                                         size_f,
                                         stride_ix_s,
                                         stride_ix_b,
                                         stride_ix_h,
                                         stride_ix_d,
                                         stride_iy_s,
                                         stride_iy_b,
                                         stride_iy_h,
                                         stride_iy_d,
                                         stride_ox_s,
                                         stride_ox_b,
                                         stride_ox_h,
                                         stride_ox_d,
                                         stride_oy_s,
                                         stride_oy_b,
                                         stride_oy_h,
                                         stride_oy_d););
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_thd_uncached(scalar_t* __restrict__ p_output,
                              const scalar_t* __restrict__ p_input,
                              const int32_t* __restrict__ p_cu_seqlens,
                              const scalar_f_t* __restrict__ p_freqs,
                              const int32_t size_max_s,
                              const int32_t size_b,
                              const int32_t size_h,
                              const int32_t size_d,
                              const int32_t size_f, // size of last dimension of freqs.
                              const int32_t stride_i_t,
                              const int32_t stride_i_h,
                              const int32_t stride_i_d,
                              const int32_t stride_o_t,
                              const int32_t stride_o_h,
                              const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] = get_grid_config<ReuseFreqsFrontPart, false>(size_max_s, 1, size_b, size_f);

    if(p_output == p_input)
    {
        assert(stride_i_t == stride_o_t);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(RotateStyle,
                                               stride_i_d,
                                               kn_entry_1c_thd_uncached_inplace<Op,
                                                                                RotateStyle,
                                                                                ReuseFreqsFrontPart,
                                                                                NopeFirst,
                                                                                Stride0Eq1>
                                               <<<grid, block, 0, stream>>>(p_output,
                                                                            p_cu_seqlens,
                                                                            p_freqs,
                                                                            size_h,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_i_t,
                                                                            stride_i_h,
                                                                            stride_i_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(RotateStyle,
                                               stride_o_d,
                                               stride_i_d,
                                               kn_entry_1c_thd_uncached<Op,
                                                                        RotateStyle,
                                                                        ReuseFreqsFrontPart,
                                                                        NopeFirst,
                                                                        Stride0Eq1,
                                                                        Stride1Eq1>
                                               <<<grid, block, 0, stream>>>(p_output,
                                                                            p_input,
                                                                            p_cu_seqlens,
                                                                            p_freqs,
                                                                            size_h,
                                                                            size_d,
                                                                            size_f,
                                                                            stride_i_t,
                                                                            stride_i_h,
                                                                            stride_i_d,
                                                                            stride_o_t,
                                                                            stride_o_h,
                                                                            stride_o_d););
    }
}

template <typename Op,
          int32_t RotateStyle,
          bool ReuseFreqsFrontPart,
          bool NopeFirst,
          typename scalar_t,
          typename scalar_f_t>
void dispatch_1c_2d_cached(scalar_t* __restrict__ p_output,
                           const scalar_t* __restrict__ p_input,
                           const scalar_f_t* __restrict__ p_cos_h,
                           const scalar_f_t* __restrict__ p_sin_h,
                           const scalar_f_t* __restrict__ p_cos_w,
                           const scalar_f_t* __restrict__ p_sin_w,
                           const int img_height,
                           const int img_width,
                           const int32_t size_b,
                           const int32_t size_h,
                           const int32_t size_d,
                           const int32_t stride_i_b,
                           const int32_t stride_i_s,
                           const int32_t stride_i_h,
                           const int32_t stride_i_d,
                           const int32_t stride_o_b,
                           const int32_t stride_o_s,
                           const int32_t stride_o_h,
                           const int32_t stride_o_d)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    auto [grid, block] =
        get_grid_config<ReuseFreqsFrontPart, true>(img_height, img_width, size_b, size_d >> 1);

    if(p_output == p_input)
    {
        assert(stride_i_s == stride_o_s);
        assert(stride_i_b == stride_o_b);
        assert(stride_i_h == stride_o_h);
        assert(stride_i_d == stride_o_d);

        LAUNCH_KERNEL_STRIDE_EQUAL_1_1_STRIDES(RotateStyle,
                                               stride_i_d,
                                               kn_entry_1c_2d_cached_inplace<Op,
                                                                             RotateStyle,
                                                                             ReuseFreqsFrontPart,
                                                                             NopeFirst,
                                                                             Stride0Eq1>
                                               <<<grid, block, 0, stream>>>(p_output,
                                                                            p_cos_h,
                                                                            p_sin_h,
                                                                            p_cos_w,
                                                                            p_sin_w,
                                                                            img_width,
                                                                            size_h,
                                                                            size_d,
                                                                            stride_i_b,
                                                                            stride_i_s,
                                                                            stride_i_h,
                                                                            stride_i_d););
    }
    else
    {
        LAUNCH_KERNEL_STRIDE_EQUAL_1_2_STRIDES(RotateStyle,
                                               stride_o_d,
                                               stride_i_d,
                                               kn_entry_1c_2d_cached<Op,
                                                                     RotateStyle,
                                                                     ReuseFreqsFrontPart,
                                                                     NopeFirst,
                                                                     Stride0Eq1,
                                                                     Stride1Eq1>
                                               <<<grid, block, 0, stream>>>(p_output,
                                                                            p_input,
                                                                            p_cos_h,
                                                                            p_sin_h,
                                                                            p_cos_w,
                                                                            p_sin_w,
                                                                            img_width,
                                                                            size_h,
                                                                            size_d,
                                                                            stride_i_b,
                                                                            stride_i_s,
                                                                            stride_i_h,
                                                                            stride_i_d,
                                                                            stride_o_b,
                                                                            stride_o_s,
                                                                            stride_o_h,
                                                                            stride_o_d););
    }
}
} // namespace aiter

#define DISPATCH_ROPE_TYPES_PARAMS(                                               \
    TYPE0, TYPE1, ROTATE_STYLE, REUSE_FREQS_FRONT_PART, NOPE_FIRST, NAME, ...)    \
    switch((TYPE0))                                                               \
    {                                                                             \
    case at::ScalarType::Float: {                                                 \
        using scalar_t_0 = float;                                                 \
        switch((TYPE1))                                                           \
        {                                                                         \
        case at::ScalarType::Float: {                                             \
            using scalar_t_1 = float;                                             \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::Half: {                                              \
            using scalar_t_1 = at::Half;                                          \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::BFloat16: {                                          \
            using scalar_t_1 = at::BFloat16;                                      \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        default:                                                                  \
            TORCH_CHECK(false,                                                    \
                        NAME " does't support ",                                  \
                        toString((TYPE0)),                                        \
                        " with ",                                                 \
                        toString((TYPE1)),                                        \
                        ".");                                                     \
        }                                                                         \
        break;                                                                    \
    }                                                                             \
    case at::ScalarType::Half: {                                                  \
        using scalar_t_0 = at::Half;                                              \
        switch((TYPE1))                                                           \
        {                                                                         \
        case at::ScalarType::Float: {                                             \
            using scalar_t_1 = float;                                             \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::Half: {                                              \
            using scalar_t_1 = at::Half;                                          \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::BFloat16: {                                          \
            using scalar_t_1 = at::BFloat16;                                      \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        default:                                                                  \
            TORCH_CHECK(false,                                                    \
                        NAME " does't support ",                                  \
                        toString((TYPE0)),                                        \
                        " with ",                                                 \
                        toString((TYPE1)),                                        \
                        ".");                                                     \
        }                                                                         \
        break;                                                                    \
    }                                                                             \
    case at::ScalarType::BFloat16: {                                              \
        using scalar_t_0 = at::BFloat16;                                          \
        switch((TYPE1))                                                           \
        {                                                                         \
        case at::ScalarType::Float: {                                             \
            using scalar_t_1 = float;                                             \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::Half: {                                              \
            using scalar_t_1 = at::Half;                                          \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        case at::ScalarType::BFloat16: {                                          \
            using scalar_t_1 = at::BFloat16;                                      \
            if((REUSE_FREQS_FRONT_PART))                                          \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = true;                        \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                constexpr bool ReuseFreqsFrontPart = false;                       \
                if((ROTATE_STYLE) == ROTATE_STYLE_NEOX)                           \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_NEOX;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else if((ROTATE_STYLE) == ROTATE_STYLE_GPTJ)                      \
                {                                                                 \
                    constexpr int32_t RotateStyle = ROTATE_STYLE_GPTJ;            \
                    if((NOPE_FIRST))                                              \
                    {                                                             \
                        constexpr bool NopeFirst = true;                          \
                        __VA_ARGS__;                                              \
                    }                                                             \
                    else                                                          \
                    {                                                             \
                        constexpr bool NopeFirst = false;                         \
                        __VA_ARGS__;                                              \
                    }                                                             \
                }                                                                 \
                else                                                              \
                {                                                                 \
                    TORCH_CHECK(false,                                            \
                                NAME " does't support rotate type ",              \
                                std::to_string((ROTATE_STYLE)),                   \
                                ".");                                             \
                }                                                                 \
            }                                                                     \
            break;                                                                \
        }                                                                         \
        default:                                                                  \
            TORCH_CHECK(false,                                                    \
                        NAME " does't support ",                                  \
                        toString((TYPE0)),                                        \
                        " with ",                                                 \
                        toString((TYPE1)),                                        \
                        ".");                                                     \
        }                                                                         \
        break;                                                                    \
    }                                                                             \
    default: TORCH_CHECK(false, NAME " does't support ", toString((TYPE0)), "."); \
    }

namespace mrope_utils {

static constexpr int kBytesPerAccess = 16;
static constexpr int WARP_SIZE       = 32;

namespace block_utils {

template <typename T>
__inline__ __device__ T warp_shfl_xor_sync(T val, int offset)
{
    return __shfl_xor(val, offset, 32);
}

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val)
{
#pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
        val += warp_shfl_xor_sync(val, offset);
    return val;
}

template <typename T>
__inline__ __device__ T warp_shfl_sync(T val, int src_id)
{
    return __shfl(val, src_id, 32);
}

} // namespace block_utils

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) vec_t
{
    T data[vec_size];
    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ T const& operator[](int i) const { return data[i]; }
    __device__ __forceinline__ void load(const T* ptr)
    {
        *this = *reinterpret_cast<vec_t<T, vec_size>*>(const_cast<T*>(ptr));
    }
    __device__ __forceinline__ void loop_load(const T* ptr)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            data[i] = ptr[i];
        }
    }
    __device__ __forceinline__ void store(T* ptr)
    {
        *reinterpret_cast<vec_t<T, vec_size>*>(ptr) = *this;
    }
    __device__ __forceinline__ void loop_store(T* ptr)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            ptr[i] = data[i];
        }
    }
    __device__ __forceinline__ void nontemporal_load(const T* ptr)
    {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for(int i = 0; i < ITERS; ++i)
        {
            reinterpret_cast<uint32_t*>(&data)[i] = __builtin_nontemporal_load((uint32_t*)ptr + i);
        }
    }
    __device__ __forceinline__ void nontemporal_store(T* ptr)
    {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for(int i = 0; i < ITERS; ++i)
        {
            __builtin_nontemporal_store(reinterpret_cast<uint32_t*>(&data)[i], (uint32_t*)ptr + i);
        }
    }
    __device__ __forceinline__ void fill(T val)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            data[i] = val;
        }
    }
    __device__ __forceinline__ void copy_(const vec_t<T, vec_size>& other)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            data[i] = other.data[i];
        }
    }
    template <typename IT>
    __device__ __forceinline__ void from_(const vec_t<IT, vec_size>& src, float scale)
    {
#pragma unroll
        for(int i = 0; i < vec_size; ++i)
        {
            if constexpr(std::is_same_v<T, IT>)
            {
                data[i] = src[i];
            }
            else
            {
                data[i] = ck_tile::type_convert<T>(ck_tile::type_convert<float>(src[i]) / scale);
            }
        }
    }
};

template <typename T, int vec_size>
__inline__ __device__ vec_t<T, vec_size> warp_shfl_sync_vec(vec_t<T, vec_size>& val, int offset)
{
    constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
    vec_t<T, vec_size> out;
#pragma unroll
    for(int i = 0; i < ITERS; ++i)
    {
        uint32_t val_                        = reinterpret_cast<uint32_t*>(&val)[i];
        reinterpret_cast<uint32_t*>(&out)[i] = block_utils::warp_shfl_sync<uint32_t>(val_, offset);
    }
    return out;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void
warp_rms_norm_(vec_t<T, VEC_SIZE>& input, vec_t<T, VEC_SIZE>& gamma, float rms_dim, float rms_eps)
{
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for(int i = 0; i < VEC_SIZE; ++i)
    {
        float v = (float)input[i];
        acc += v * v;
    }
    int warp_id   = threadIdx.x / 32;
    int warp_t_id = threadIdx.x % 32;
    acc           = block_utils::warp_reduce_sum<float>(acc);
    acc           = block_utils::warp_shfl_sync<float>(acc, 0);
    auto s_val    = rsqrtf(acc / rms_dim + rms_eps);
#pragma unroll
    for(int i = 0; i < VEC_SIZE; ++i)
    {
        input[i] = static_cast<T>((float)input[i] * s_val * (float)gamma[i]);
    }
}

template <typename T, int VEC_SIZE, int HEAD_SIZE, bool IS_INTERLEAVED, int M>
__device__ __forceinline__ void mrope_load_cos_sin_vec(vec_t<T, VEC_SIZE>& out,
                                                       const T* cos_sin,
                                                       const int64_t* positions,
                                                       int64_t ps0,
                                                       int64_t ps1,
                                                       int64_t token_id,
                                                       int64_t num_tokens,
                                                       int access_id_in_head,
                                                       std::array<int64_t, M>& mrope_section)
{
    constexpr int HALF_HEAD_SIZE = HEAD_SIZE / 2;
    if constexpr(IS_INTERLEAVED)
    {
        for(int i = 0; i < VEC_SIZE; ++i)
        {
            auto id   = access_id_in_head + i;
            auto id_  = (access_id_in_head < HALF_HEAD_SIZE) ? id : id - HALF_HEAD_SIZE;
            auto mid_ = id_ % M;
            if(mid_ >= 1 && id_ < mrope_section[mid_] * M)
            {
                auto p = positions[mid_ * ps0 + token_id * ps1];
                out[i] = cos_sin[p * HEAD_SIZE + id];
            }
            else
            {
                out[i] = cos_sin[positions[token_id * ps1] * HEAD_SIZE + id];
            }
        }
    }
    else
    {
        for(int i = 0; i < VEC_SIZE; ++i)
        {
            auto id  = access_id_in_head + i;
            auto id_ = (access_id_in_head < HALF_HEAD_SIZE) ? id : id - HALF_HEAD_SIZE;
            int mid;
            int end = 0;
            for(mid = 0; mid < M; ++mid)
            {
                end += mrope_section[mid];
                if(id_ < end)
                    break;
            }
            auto p = positions[mid * ps0 + token_id * ps1];
            out[i] = cos_sin[p * HEAD_SIZE + id];
        }
    }
}

struct alignas(1) fp8e4m3fn
{
    struct from_bits_t
    {
    };
    __host__ __device__ static constexpr from_bits_t from_bits() { return from_bits_t(); }
    uint8_t data;

    fp8e4m3fn()                                               = default;
    __host__ __device__ constexpr fp8e4m3fn(const fp8e4m3fn&) = default;
    __host__ __device__ constexpr fp8e4m3fn(uint8_t v)        = delete;
    explicit __host__ __device__ constexpr fp8e4m3fn(uint8_t v, from_bits_t) : data(v) {}

    explicit __host__ __device__ fp8e4m3fn(float v)
    {
        data = hip_fp8_impl::to_float8<4, 3, float, false /*negative_zero_nan*/, true /*clip*/>(v);
    }

    explicit __host__ __device__ fp8e4m3fn(double v) : fp8e4m3fn(static_cast<float>(v)) {}

    explicit inline __host__ __device__ operator float() const
    {
        return hip_fp8_impl::from_float8<4, 3, float, false /*negative_zero_nan*/>(data);
    }
};

struct alignas(1) fp8e4m3fnuz
{
    struct from_bits_t
    {
    };
    __host__ __device__ static constexpr from_bits_t from_bits() { return from_bits_t(); }
    uint8_t data;

    fp8e4m3fnuz()                                                 = default;
    __host__ __device__ constexpr fp8e4m3fnuz(const fp8e4m3fnuz&) = default;
    __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v)          = delete;
    explicit __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v, from_bits_t) : data(v) {}

    explicit __host__ __device__ fp8e4m3fnuz(float v)
    {
        data = hip_fp8_impl::to_float8<4, 3, float, true /*negative_zero_nan*/, true /*clip*/>(v);
    }

    explicit __host__ __device__ fp8e4m3fnuz(double v) : fp8e4m3fnuz(static_cast<float>(v)) {}

    explicit inline __host__ __device__ operator float() const
    {
        return hip_fp8_impl::from_float8<4, 3, float, true /*negative_zero_nan*/>(data);
    }
};

template <int HEAD_SIZE>
__device__ __forceinline__ int64_t get_shuffle_layout_k_base(const int64_t slot_id,
                                                             const int block_size,
                                                             const int num_heads_k,
                                                             const int head_id_k,
                                                             const int access_id_in_head,
                                                             const int x)
{
    // Shuffle layout: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    const int block_id      = static_cast<int>(slot_id / block_size);
    const int block_offset  = static_cast<int>(slot_id % block_size);
    const int k_head_stride = HEAD_SIZE * block_size;
    const int64_t dst_base =
        static_cast<int64_t>(block_id) * num_heads_k * k_head_stride + head_id_k * k_head_stride;
    // Pre-compute K base offset: since VEC_SIZE <= x, all elements are in the same
    // chunk
    const int chunk_id     = access_id_in_head / x;
    const int block_size_x = block_size * x;
    const int64_t k_base =
        dst_base + chunk_id * block_size_x + block_offset * x + (access_id_in_head % x);
    return k_base;
}

template <int HEAD_SIZE>
__device__ __forceinline__ int64_t get_shuffle_layout_v_base(const int64_t slot_id,
                                                             const int block_size,
                                                             const int num_heads_v,
                                                             const int head_id_v,
                                                             const int access_id_in_head,
                                                             const int x)
{
    // Shuffle layout: [num_blocks, num_kv_heads, block_size // x, head_size, x]
    const int block_id      = static_cast<int>(slot_id / block_size);
    const int block_offset  = static_cast<int>(slot_id % block_size);
    const int v_head_stride = (block_size / x) * HEAD_SIZE * x;
    const int64_t dst_base =
        static_cast<int64_t>(block_id) * num_heads_v * v_head_stride + head_id_v * v_head_stride;
    // Pre-compute V base offset (fixed for this token)
    const int v_slot_chunk    = block_offset / x;
    const int v_slot_in_chunk = block_offset % x;
    const int64_t v_base      = dst_base + v_slot_chunk * HEAD_SIZE * x + v_slot_in_chunk;
    return v_base;
}

template <typename T,
          int HEAD_SIZE,
          bool IS_NEOX,
          bool IS_MROPE,
          bool IS_INTERLEAVED,
          int M,
          typename KVT>
__global__ void fused_mrope_rms_kv_kernel(const T* qkv,
                                          const T* q_w,
                                          const T* k_w,
                                          const T* cos_sin,
                                          const int64_t* positions,
                                          int64_t positions_stride_0,
                                          int64_t positions_stride_1,
                                          int num_heads_q,
                                          int num_heads_k,
                                          int num_heads_v,
                                          double eps,
                                          std::array<int64_t, M> mrope_section,
                                          int num_tokens,
                                          int total_warps,
                                          T* q_out,
                                          KVT* k_cache,
                                          KVT* v_cache,
                                          const int64_t* slot_mapping,
                                          float per_tensor_k_scale = 1.0,
                                          float per_tensor_v_scale = 1.0,
                                          KVT* k_out               = nullptr,
                                          KVT* v_out               = nullptr,
                                          bool use_shuffle_layout  = false,
                                          int block_size           = 0,
                                          int x                    = 0)
{
    constexpr int VEC_SIZE        = HEAD_SIZE / WARP_SIZE;
    constexpr int HALF_HEAD_SIZE  = HEAD_SIZE / 2;
    const int warp_id             = threadIdx.x / WARP_SIZE;
    const int num_warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id      = blockIdx.x * num_warps_per_block + warp_id;

    if(global_warp_id >= total_warps)
        return;

    // Warp allocation: all Q first, then all K, then all V
    const int num_heads_qk   = num_heads_q + num_heads_k;
    const int num_heads      = num_heads_q + num_heads_k + num_heads_v;
    const int total_q_warps  = num_tokens * num_heads_q;
    const int total_k_warps  = num_tokens * num_heads_k;
    const int total_qk_warps = total_q_warps + total_k_warps;

    // Determine if current warp processes Q, K, or V
    const bool is_q = global_warp_id < total_q_warps;
    const bool is_k = !is_q && global_warp_id < total_qk_warps;
    const bool is_v = global_warp_id >= total_qk_warps;

    int token_id, head_id_in_token;

    if(is_q)
    {
        // Q warps: global_warp_id in range [0, total_q_warps)
        token_id         = global_warp_id / num_heads_q;
        head_id_in_token = global_warp_id % num_heads_q;
    }
    else if(is_k)
    {
        // K warps: global_warp_id in range [total_q_warps, total_qk_warps)
        const int k_warp_id = global_warp_id - total_q_warps;
        token_id            = k_warp_id / num_heads_k;
        head_id_in_token    = num_heads_q + (k_warp_id % num_heads_k);
    }
    else
    {
        // V warps: global_warp_id in range [total_qk_warps, total_warps)
        const int v_warp_id = global_warp_id - total_qk_warps;
        token_id            = v_warp_id / num_heads_v;
        head_id_in_token    = num_heads_qk + (v_warp_id % num_heads_v);
    }

    const int access_id_in_head = (threadIdx.x % WARP_SIZE) * VEC_SIZE;
    const int neighbor_offset =
        access_id_in_head < HALF_HEAD_SIZE ? HALF_HEAD_SIZE / VEC_SIZE : -HALF_HEAD_SIZE / VEC_SIZE;
    const T* qkv_ =
        &qkv[(static_cast<int64_t>(token_id) * num_heads + head_id_in_token) * HEAD_SIZE];

    if(!is_v)
    {
        vec_t<T, VEC_SIZE> w_vec;
        if(is_q)
        {
            w_vec.load(q_w + access_id_in_head);
        }
        else
        {
            w_vec.load(k_w + access_id_in_head);
        }
        vec_t<T, VEC_SIZE> x_vec;
        x_vec.load(qkv_ + access_id_in_head);
        vec_t<T, VEC_SIZE> out_vec;
        if constexpr(IS_NEOX)
        {
            vec_t<T, VEC_SIZE> cos_sin_vec;
            if constexpr(IS_MROPE)
            {
                mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
                    cos_sin_vec,
                    cos_sin,
                    positions,
                    positions_stride_0,
                    positions_stride_1,
                    token_id,
                    num_tokens,
                    access_id_in_head,
                    mrope_section);
            }
            else
            {
                auto position_ = positions[token_id * positions_stride_1];
                cos_sin_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head]);
            }
            warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
            auto nb_cos_sin_vec =
                warp_shfl_sync_vec<T, VEC_SIZE>(cos_sin_vec, threadIdx.x + neighbor_offset);
            auto nb_x_vec = warp_shfl_sync_vec<T, VEC_SIZE>(x_vec, threadIdx.x + neighbor_offset);
            if(neighbor_offset > 0)
            {
#pragma unroll
                for(int i = 0; i < VEC_SIZE; ++i)
                {
                    out_vec[i] =
                        (float)x_vec[i] * (float)cos_sin_vec[i] -
                        (float)nb_x_vec[i] * (float)nb_cos_sin_vec[i]; // x0 * cos - x1 * sin
                }
            }
            else
            {
#pragma unroll
                for(int i = 0; i < VEC_SIZE; ++i)
                {
                    out_vec[i] = (float)x_vec[i] * (float)nb_cos_sin_vec[i] +
                                 (float)nb_x_vec[i] * (float)cos_sin_vec[i]; // x1 * cos + x0 * sin
                }
            }
        }
        else
        {
            vec_t<T, VEC_SIZE> cos_vec, sin_vec;
            if constexpr(IS_MROPE)
            {
                mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
                    cos_vec,
                    cos_sin,
                    positions,
                    positions_stride_0,
                    positions_stride_1,
                    token_id,
                    num_tokens,
                    access_id_in_head / 2,
                    mrope_section);
                mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
                    sin_vec,
                    cos_sin,
                    positions,
                    positions_stride_0,
                    positions_stride_1,
                    token_id,
                    num_tokens,
                    access_id_in_head / 2 + HALF_HEAD_SIZE,
                    mrope_section);
            }
            else
            {
                auto position_ = positions[token_id * positions_stride_1];
                cos_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2]);
                sin_vec.load(
                    &cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE]);
            }
            warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
#pragma unroll
            for(int i = 0; i < VEC_SIZE / 2; ++i)
            {
                out_vec[2 * i + 0] = (float)x_vec[2 * i + 0] * (float)cos_vec[i] -
                                     (float)x_vec[2 * i + 1] * (float)sin_vec[i];
                out_vec[2 * i + 1] = (float)x_vec[2 * i + 1] * (float)cos_vec[i] +
                                     (float)x_vec[2 * i + 0] * (float)sin_vec[i];
            }
        }

        if(is_q)
        {
            T* q_ = &q_out[(static_cast<int64_t>(token_id) * num_heads_q + head_id_in_token) *
                           HEAD_SIZE];
            out_vec.store(q_ + access_id_in_head);
        }
        else
        {
            vec_t<KVT, VEC_SIZE> out_kv_vec;
            out_kv_vec.from_(out_vec, per_tensor_k_scale);
            const int64_t slot_id = slot_mapping[token_id];
            if(slot_id < 0)
                return;
            const int head_id_k = head_id_in_token - num_heads_q;
            if(use_shuffle_layout)
            {
                int64_t k_base = get_shuffle_layout_k_base<HEAD_SIZE>(
                    slot_id, block_size, num_heads_k, head_id_k, access_id_in_head, x);
                out_kv_vec.store(k_cache + k_base);
            }
            else
            {
                const int64_t offset =
                    (slot_id * num_heads_k + head_id_k) * HEAD_SIZE + access_id_in_head;
                out_kv_vec.store(k_cache + offset);
            }
            if(k_out != nullptr)
            {
                const int64_t k_out_offset =
                    (static_cast<int64_t>(token_id) * num_heads_k + head_id_k) * HEAD_SIZE +
                    access_id_in_head;
                out_kv_vec.store(k_out + k_out_offset);
            }
        }
    }
    else
    {
        vec_t<T, VEC_SIZE> out_vec;
        vec_t<KVT, VEC_SIZE> out_kv_vec;
        out_vec.load(qkv_ + access_id_in_head);
        out_kv_vec.from_(out_vec, per_tensor_v_scale);
        const int64_t slot_id = slot_mapping[token_id];
        if(slot_id < 0)
            return;
        const int head_id_v = head_id_in_token - num_heads_qk;
        if(use_shuffle_layout)
        {
            int64_t v_base = get_shuffle_layout_v_base<HEAD_SIZE>(
                slot_id, block_size, num_heads_v, head_id_v, access_id_in_head, x);
#pragma unroll
            for(int i = 0; i < VEC_SIZE; ++i)
            {
                const int offset_in_head             = access_id_in_head + i;
                v_cache[v_base + offset_in_head * x] = out_kv_vec[i];
            }
        }
        else
        {
            const int64_t offset =
                (slot_id * num_heads_v + head_id_v) * HEAD_SIZE + access_id_in_head;
            out_kv_vec.store(v_cache + offset);
        }
        if(v_out != nullptr)
        {
            const int64_t v_out_offset =
                (static_cast<int64_t>(token_id) * num_heads_v + head_id_v) * HEAD_SIZE +
                access_id_in_head;
            out_kv_vec.store(v_out + v_out_offset);
        }
    }
}

template <typename T, int M, typename KVT>
void fused_mrope_rms_set_kv(const T* qkv,
                            const T* q_w,
                            const T* k_w,
                            const T* cos_sin,
                            const int64_t* positions,
                            int64_t positions_stride_0,
                            int64_t positions_stride_1,
                            int64_t num_tokens,
                            int64_t num_heads_q,
                            int64_t num_heads_k,
                            int64_t num_heads_v,
                            int64_t head_size,
                            bool is_neox_style,
                            double eps,
                            std::array<int64_t, M> mrope_section,
                            bool is_interleaved,
                            T* q_out,
                            KVT* k_cache,
                            KVT* v_cache,
                            const int64_t* slot_mapping,
                            hipStream_t stream,
                            float per_tensor_k_scale = 1.0,
                            float per_tensor_v_scale = 1.0,
                            KVT* k_out               = nullptr,
                            KVT* v_out               = nullptr,
                            bool use_shuffle_layout  = false,
                            int64_t block_size       = 0,
                            int64_t x                = 0)
{
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    auto dim = std::accumulate(mrope_section.begin(), mrope_section.end(), 0);
    TORCH_CHECK(dim == head_size / 2);
    constexpr int THREAD_BLOCK_SIZE = 256;
    auto total_warps                = num_tokens * (num_heads_q + num_heads_k + num_heads_v);
    auto num_warps_per_block        = THREAD_BLOCK_SIZE / WARP_SIZE;
    dim3 threadsPerBlock(THREAD_BLOCK_SIZE);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block);

#define DISPATCH_NEOX(HEAD_SIZE, IS_INTERLEAVED)                                     \
    if(is_neox_style)                                                                \
    {                                                                                \
        fused_mrope_rms_kv_kernel<T, HEAD_SIZE, true, true, IS_INTERLEAVED, M, KVT>  \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                         \
                                                        q_w,                         \
                                                        k_w,                         \
                                                        cos_sin,                     \
                                                        positions,                   \
                                                        positions_stride_0,          \
                                                        positions_stride_1,          \
                                                        num_heads_q,                 \
                                                        num_heads_k,                 \
                                                        num_heads_v,                 \
                                                        eps,                         \
                                                        mrope_section,               \
                                                        num_tokens,                  \
                                                        total_warps,                 \
                                                        q_out,                       \
                                                        k_cache,                     \
                                                        v_cache,                     \
                                                        slot_mapping,                \
                                                        per_tensor_k_scale,          \
                                                        per_tensor_v_scale,          \
                                                        k_out,                       \
                                                        v_out,                       \
                                                        use_shuffle_layout,          \
                                                        block_size,                  \
                                                        x);                          \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        fused_mrope_rms_kv_kernel<T, HEAD_SIZE, false, true, IS_INTERLEAVED, M, KVT> \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                         \
                                                        q_w,                         \
                                                        k_w,                         \
                                                        cos_sin,                     \
                                                        positions,                   \
                                                        positions_stride_0,          \
                                                        positions_stride_1,          \
                                                        num_heads_q,                 \
                                                        num_heads_k,                 \
                                                        num_heads_v,                 \
                                                        eps,                         \
                                                        mrope_section,               \
                                                        num_tokens,                  \
                                                        total_warps,                 \
                                                        q_out,                       \
                                                        k_cache,                     \
                                                        v_cache,                     \
                                                        slot_mapping,                \
                                                        per_tensor_k_scale,          \
                                                        per_tensor_v_scale,          \
                                                        k_out,                       \
                                                        v_out,                       \
                                                        use_shuffle_layout,          \
                                                        block_size,                  \
                                                        x);                          \
    }

    if(is_interleaved)
    {
        switch(head_size)
        {
        case 64: DISPATCH_NEOX(64, true) break;
        case 128: DISPATCH_NEOX(128, true) break;
        case 256: DISPATCH_NEOX(256, true) break;
        }
    }
    else
    {
        switch(head_size)
        {
        case 64: DISPATCH_NEOX(64, false) break;
        case 128: DISPATCH_NEOX(128, false) break;
        case 256: DISPATCH_NEOX(256, false) break;
        }
    }

#undef DISPATCH_NEOX
}

template <typename T, typename KVT>
void fused_rope_rms_set_kv(const T* qkv,
                           const T* q_w,
                           const T* k_w,
                           const T* cos_sin,
                           const int64_t* positions,
                           int64_t positions_stride_0,
                           int64_t positions_stride_1,
                           int64_t num_tokens,
                           int64_t num_heads_q,
                           int64_t num_heads_k,
                           int64_t num_heads_v,
                           int64_t head_size,
                           bool is_neox_style,
                           double eps,
                           T* q_out,
                           KVT* k_cache,
                           KVT* v_cache,
                           const int64_t* slot_mapping,
                           hipStream_t stream,
                           float per_tensor_k_scale = 1.0,
                           float per_tensor_v_scale = 1.0,
                           KVT* k_out               = nullptr,
                           KVT* v_out               = nullptr,
                           bool use_shuffle_layout  = false,
                           int64_t block_size       = 0,
                           int64_t x                = 0)
{
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    constexpr int THREAD_BLOCK_SIZE = 256;
    auto total_warps                = num_tokens * (num_heads_q + num_heads_k + num_heads_v);
    auto num_warps_per_block        = THREAD_BLOCK_SIZE / WARP_SIZE;
    dim3 threadsPerBlock(THREAD_BLOCK_SIZE);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block);
    std::array<int64_t, 1> mrope_section = {0};

#define DISPATCH_NEOX(HEAD_SIZE)                                             \
    if(is_neox_style)                                                        \
    {                                                                        \
        fused_mrope_rms_kv_kernel<T, HEAD_SIZE, true, false, false, 1, KVT>  \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                 \
                                                        q_w,                 \
                                                        k_w,                 \
                                                        cos_sin,             \
                                                        positions,           \
                                                        positions_stride_0,  \
                                                        positions_stride_1,  \
                                                        num_heads_q,         \
                                                        num_heads_k,         \
                                                        num_heads_v,         \
                                                        eps,                 \
                                                        mrope_section,       \
                                                        num_tokens,          \
                                                        total_warps,         \
                                                        q_out,               \
                                                        k_cache,             \
                                                        v_cache,             \
                                                        slot_mapping,        \
                                                        per_tensor_k_scale,  \
                                                        per_tensor_v_scale,  \
                                                        k_out,               \
                                                        v_out,               \
                                                        use_shuffle_layout,  \
                                                        block_size,          \
                                                        x);                  \
    }                                                                        \
    else                                                                     \
    {                                                                        \
        fused_mrope_rms_kv_kernel<T, HEAD_SIZE, false, false, false, 1, KVT> \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                 \
                                                        q_w,                 \
                                                        k_w,                 \
                                                        cos_sin,             \
                                                        positions,           \
                                                        positions_stride_0,  \
                                                        positions_stride_1,  \
                                                        num_heads_q,         \
                                                        num_heads_k,         \
                                                        num_heads_v,         \
                                                        eps,                 \
                                                        mrope_section,       \
                                                        num_tokens,          \
                                                        total_warps,         \
                                                        q_out,               \
                                                        k_cache,             \
                                                        v_cache,             \
                                                        slot_mapping,        \
                                                        per_tensor_k_scale,  \
                                                        per_tensor_v_scale,  \
                                                        k_out,               \
                                                        v_out,               \
                                                        use_shuffle_layout,  \
                                                        block_size,          \
                                                        x);                  \
    }

    switch(head_size)
    {
    case 64: DISPATCH_NEOX(64) break;
    case 128: DISPATCH_NEOX(128) break;
    case 256: DISPATCH_NEOX(256) break;
    }

#undef DISPATCH_NEOX
}

} // namespace mrope_utils
