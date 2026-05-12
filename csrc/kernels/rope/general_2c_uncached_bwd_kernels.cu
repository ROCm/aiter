// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"
using namespace aiter;

void rope_2c_bwd_impl(
    aiter_tensor_t&       input_grads_x, // [s, b, h, d]
    aiter_tensor_t&       input_grads_y, // [s, b, h, d]
    const aiter_tensor_t& output_grads_x,// [s, b, h, d]
    const aiter_tensor_t& output_grads_y,// [s, b, h, d]
    const aiter_tensor_t& freqs,         // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    const int32_t size_s   = output_grads_x.size(0);
    const int32_t size_b   = output_grads_x.size(1);
    const int32_t size_h_x = output_grads_x.size(2);
    const int32_t size_h_y = output_grads_y.size(2);
    const int32_t size_d   = output_grads_x.size(3);
    const int32_t size_f   = freqs.size(3);
    const int32_t stride_ox_s = output_grads_x.stride(0);
    const int32_t stride_ox_b = output_grads_x.stride(1);
    const int32_t stride_ox_h = output_grads_x.stride(2);
    const int32_t stride_ox_d = output_grads_x.stride(3);
    const int32_t stride_oy_s = output_grads_y.stride(0);
    const int32_t stride_oy_b = output_grads_y.stride(1);
    const int32_t stride_oy_h = output_grads_y.stride(2);
    const int32_t stride_oy_d = output_grads_y.stride(3);
    const int32_t stride_ix_s = input_grads_x.stride(0);
    const int32_t stride_ix_b = input_grads_x.stride(1);
    const int32_t stride_ix_h = input_grads_x.stride(2);
    const int32_t stride_ix_d = input_grads_x.stride(3);
    const int32_t stride_iy_s = input_grads_y.stride(0);
    const int32_t stride_iy_b = input_grads_y.stride(1);
    const int32_t stride_iy_h = input_grads_y.stride(2);
    const int32_t stride_iy_d = input_grads_y.stride(3);

    AITER_CHECK(stride_ix_d == 1 && stride_iy_d == 1 && stride_ox_d == 1 && stride_oy_d == 1,
                "rope_2c_bwd_impl requires all stride_d to be 1");

    HipDeviceGuard device_guard(input_grads_x.device_id);
    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads_x.dtype(),
        freqs.dtype(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_2c_sbhd_uncached<OpUncachedBwd, ...>",
        dispatch_2c_sbhd_uncached<OpUncachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst, true>(
            reinterpret_cast<scalar_t_0*>(input_grads_x.data_ptr()),
            reinterpret_cast<scalar_t_0*>(input_grads_y.data_ptr()),
            reinterpret_cast<scalar_t_0*>(output_grads_x.data_ptr()),
            reinterpret_cast<scalar_t_0*>(output_grads_y.data_ptr()),
            reinterpret_cast<scalar_t_1*>(freqs.data_ptr()),
            size_s, size_b, size_h_x, size_h_y, size_d,
            size_f,
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d,
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d););
}
