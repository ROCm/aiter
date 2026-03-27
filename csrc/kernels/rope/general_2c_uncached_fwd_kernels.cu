// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"
using namespace aiter;

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
    const int32_t size_s   = input_x.size(0);
    const int32_t size_b   = input_x.size(1);
    const int32_t size_h_x = input_x.size(2);
    const int32_t size_h_y = input_y.size(2);
    const int32_t size_d   = input_x.size(3);
    const int32_t size_f   = freqs.size(3);
    const int32_t stride_ix_s = input_x.stride(0);
    const int32_t stride_ix_b = input_x.stride(1);
    const int32_t stride_ix_h = input_x.stride(2);
    const int32_t stride_ix_d = input_x.stride(3);
    const int32_t stride_iy_s = input_y.stride(0);
    const int32_t stride_iy_b = input_y.stride(1);
    const int32_t stride_iy_h = input_y.stride(2);
    const int32_t stride_iy_d = input_y.stride(3);
    const int32_t stride_ox_s = output_x.stride(0);
    const int32_t stride_ox_b = output_x.stride(1);
    const int32_t stride_ox_h = output_x.stride(2);
    const int32_t stride_ox_d = output_x.stride(3);
    const int32_t stride_oy_s = output_y.stride(0);
    const int32_t stride_oy_b = output_y.stride(1);
    const int32_t stride_oy_h = output_y.stride(2);
    const int32_t stride_oy_d = output_y.stride(3);

    TORCH_CHECK(stride_ix_d == 1 && stride_iy_d == 1 && stride_ox_d == 1 && stride_oy_d == 1,
                "rope_2c_fwd_impl requires all stride_d to be 1");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input_x));
    DISPATCH_ROPE_TYPES_PARAMS(
        input_x.scalar_type(),
        freqs.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_2c_sbhd_uncached<OpUncachedFwd, ...>",
        dispatch_2c_sbhd_uncached<OpUncachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst, true>(
            output_x.data_ptr<scalar_t_0>(),
            output_y.data_ptr<scalar_t_0>(),
            input_x.data_ptr<scalar_t_0>(),
            input_y.data_ptr<scalar_t_0>(),
            freqs.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h_x, size_h_y, size_d,
            size_f,
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d););
}
