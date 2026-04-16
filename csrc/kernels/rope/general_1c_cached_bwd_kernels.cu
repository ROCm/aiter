// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"
using namespace aiter;

void rope_cached_bwd_impl(
    torch::Tensor&       input_grads,   // [s, b, h, d]
    const torch::Tensor& output_grads,  // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin,           // [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    const int32_t size_s = output_grads.size(0);
    const int32_t size_b = output_grads.size(1);
    const int32_t size_h = output_grads.size(2);
    const int32_t size_d = output_grads.size(3);
    const int32_t size_f = cos.size(3);
    const int32_t stride_o_s = output_grads.stride(0);
    const int32_t stride_o_b = output_grads.stride(1);
    const int32_t stride_o_h = output_grads.stride(2);
    const int32_t stride_o_d = output_grads.stride(3);
    const int32_t stride_i_s = input_grads.stride(0);
    const int32_t stride_i_b = input_grads.stride(1);
    const int32_t stride_i_h = input_grads.stride(2);
    const int32_t stride_i_d = input_grads.stride(3);

    TORCH_CHECK(stride_i_d == 1 && stride_o_d == 1,
                "rope_cached_bwd_impl requires all stride_d to be 1");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input_grads));
    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_1c_sbhd_cached<OpCachedBwd, ...>",
        dispatch_1c_sbhd_cached<OpCachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst, true>(
            input_grads.data_ptr<scalar_t_0>(),
            output_grads.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            size_s, size_b, size_h, size_d,
            size_f,
            stride_o_s, stride_o_b, stride_o_h, stride_o_d,
            stride_i_s, stride_i_b, stride_i_h, stride_i_d););
}
