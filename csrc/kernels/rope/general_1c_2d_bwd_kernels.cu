// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"
using namespace aiter;

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
    const int size_b = output_grads.size(0);
    const int size_s = output_grads.size(1);
    const int size_h = output_grads.size(2);
    const int size_d = output_grads.size(3);
    const int stride_o_b = output_grads.stride(0);
    const int stride_o_s = output_grads.stride(1);
    const int stride_o_h = output_grads.stride(2);
    const int stride_o_d = output_grads.stride(3);
    const int stride_i_b = input_grads.stride(0);
    const int stride_i_s = input_grads.stride(1);
    const int stride_i_h = input_grads.stride(2);
    const int stride_i_d = input_grads.stride(3);

    TORCH_CHECK(size_s == img_height * img_width, "rope_2d_bwd_impl - input tensor shape doesn't match image size.");
    TORCH_CHECK(stride_i_d == 1 && stride_o_d == 1,
                "rope_2d_bwd_impl requires all stride_d to be 1");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input_grads));
    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads.scalar_type(),
        cos_h.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_1c_2d_cached<OpCachedBwd, ...>",
        dispatch_1c_2d_cached<OpCachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst, true>(
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
