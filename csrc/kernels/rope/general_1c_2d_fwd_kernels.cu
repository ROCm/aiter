// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"
using namespace aiter;

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
    const int size_b = input.size(0);
    const int size_s = input.size(1);
    const int size_h = input.size(2);
    const int size_d = input.size(3);
    const int stride_i_b = input.stride(0);
    const int stride_i_s = input.stride(1);
    const int stride_i_h = input.stride(2);
    const int stride_i_d = input.stride(3);
    const int stride_o_b = output.stride(0);
    const int stride_o_s = output.stride(1);
    const int stride_o_h = output.stride(2);
    const int stride_o_d = output.stride(3);

    TORCH_CHECK(size_s == img_height * img_width, "rope_2d_fwd_impl - input tensor shape doesn't match image size.");
    TORCH_CHECK(stride_i_d == 1 && stride_o_d == 1,
                "rope_2d_fwd_impl requires all stride_d to be 1");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    DISPATCH_ROPE_TYPES_PARAMS(
        input.scalar_type(),
        cos_h.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_1c_2d_cached<OpCachedFwd, ...>",
        dispatch_1c_2d_cached<OpCachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst, true>(
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
