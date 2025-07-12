// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"

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

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
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

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_x));
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

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
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

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_x));
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

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
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

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
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