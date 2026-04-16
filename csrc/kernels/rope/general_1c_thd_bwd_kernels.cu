// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"
using namespace aiter;

void rope_thd_bwd_impl(
    torch::Tensor&       input_grads,   // [t, h, d]
    const torch::Tensor& output_grads,  // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [b + 1]
    const torch::Tensor& freqs,         // [max_s, 1, 1, d]
    const int            rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    const int32_t size_h     = output_grads.size(1);
    const int32_t size_d     = output_grads.size(2);
    const int32_t size_f     = freqs.size(3);
    const int32_t size_b     = cu_seqlens.size(0) - 1;
    const int32_t size_max_s = freqs.size(0);
    const int32_t stride_o_t = output_grads.stride(0);
    const int32_t stride_o_h = output_grads.stride(1);
    const int32_t stride_o_d = output_grads.stride(2);
    const int32_t stride_i_t = input_grads.stride(0);
    const int32_t stride_i_h = input_grads.stride(1);
    const int32_t stride_i_d = input_grads.stride(2);

    TORCH_CHECK(stride_i_d == 1 && stride_o_d == 1,
                "rope_thd_bwd_impl requires all stride_d to be 1");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input_grads));
    DISPATCH_ROPE_TYPES_PARAMS(
        output_grads.scalar_type(),
        freqs.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_1c_thd_uncached<OpUncachedBwd, ...>",
        dispatch_1c_thd_uncached<OpUncachedBwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst, true>(
            input_grads.data_ptr<scalar_t_0>(),
            output_grads.data_ptr<scalar_t_0>(),
            cu_seqlens.data_ptr<int32_t>(),
            freqs.data_ptr<scalar_t_1>(),
            size_max_s, size_b, size_h, size_d,
            size_f,
            stride_o_t, stride_o_h, stride_o_d,
            stride_i_t, stride_i_h, stride_i_d););
}
