// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"
using namespace aiter;

void rope_thd_fwd_impl(
    aiter_tensor_t&       output,        // [t, h, d]
    const aiter_tensor_t& input,         // [t, h, d]
    const aiter_tensor_t& cu_seqlens,    // [b + 1]
    const aiter_tensor_t& freqs,         // [max_s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    const int32_t size_h     = input.size(1);
    const int32_t size_d     = input.size(2);
    const int32_t size_f     = freqs.size(3);
    const int32_t size_b     = cu_seqlens.size(0) - 1;
    const int32_t size_max_s = freqs.size(0);
    const int32_t stride_i_t = input.stride(0);
    const int32_t stride_i_h = input.stride(1);
    const int32_t stride_i_d = input.stride(2);
    const int32_t stride_o_t = output.stride(0);
    const int32_t stride_o_h = output.stride(1);
    const int32_t stride_o_d = output.stride(2);

    AITER_CHECK(stride_i_d == 1 && stride_o_d == 1,
                "rope_thd_fwd_impl requires all stride_d to be 1");

    HipDeviceGuard device_guard(input.device_id);
    DISPATCH_ROPE_TYPES_PARAMS(
        input.dtype(),
        freqs.dtype(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_1c_thd_uncached<OpUncachedFwd, ...>",
        dispatch_1c_thd_uncached<OpUncachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst, true>(
            reinterpret_cast<scalar_t_0*>(output.data_ptr()),
            reinterpret_cast<scalar_t_0*>(input.data_ptr()),
            reinterpret_cast<int32_t*>(cu_seqlens.data_ptr()),
            reinterpret_cast<scalar_t_1*>(freqs.data_ptr()),
            size_max_s, size_b, size_h, size_d,
            size_f,
            stride_i_t, stride_i_h, stride_i_d,
            stride_o_t, stride_o_h, stride_o_d););
}
