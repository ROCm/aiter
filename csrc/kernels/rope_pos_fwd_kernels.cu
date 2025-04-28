// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"

// =====================================================================================================================
// Interfaces
//

// Parameters:
//   output:       [s, b, h, d]
//   input:        [s, b, h, d]
//   cos:          [max_pos, 1, 1, d // 2] if reuse_freqs_front_part else [max_pos, 1, 1, d]
//   sin:          [max_pos, 1, 1, d // 2] if reuse_freqs_front_part else [max_pos, 1, 1, d]
//   positions:    [s, b]
//   rotate_style: 0: NEOX style, 1: GPT-J style
void rope_cached_positions_fwd_impl(
    torch::Tensor&       output,
    const torch::Tensor& input,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& positions,
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    // Get sizes of input and output
    const int32_t size_s = min(input.size(0), positions.size(0));
    const int32_t size_b = min(input.size(1), positions.size(1));
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

    // Get strides of positions and offsets
    assert(1 == positions.stride(1) && 2 == positions.dim());
    const int32_t max_position = cos.size(0);

    DISPATCH_ROPE_TYPES_PARAMS(
        input.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_1c_sbhd_cached_indirect<OpCachedFwd, ...>",
        dispatch_1c_sbhd_cached_indirect<OpCachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
            output.data_ptr<scalar_t_0>(),
            input.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            positions.data_ptr<int64_t>(),
            max_position,
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_i_s, stride_i_b, stride_i_h, stride_i_d,
            stride_o_s, stride_o_b, stride_o_h, stride_o_d););
}

// Parameters:
//   output_x/y:   [s, b, h, d]
//   input_x/y:    [s, b, h, d]
//   cos:          [max_pos, 1, 1, d // 2] if reuse_freqs_front_part else [max_pos, 1, 1, d]
//   sin:          [max_pos, 1, 1, d // 2] if reuse_freqs_front_part else [max_pos, 1, 1, d]
//   positions:    [s, b]
//   rotate_style: 0: NEOX style, 1: GPT-J style
void rope_cached_positions_2c_fwd_impl(
    torch::Tensor&       output_x,
    torch::Tensor&       output_y,
    const torch::Tensor& input_x,
    const torch::Tensor& input_y,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& positions,
    const int32_t        rotate_style,
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
    const int32_t max_position = cos.size(0);

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
            max_position,
            size_s, size_b, size_h_x, size_h_y, size_d,
            size_f, // size of last dimension of freqs.
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d););
}

// Parameters:
//   output:       [s, b, h, d]
//   input:        [s, b, h, d]
//   cos:          [max_pos, 1, 1, d // 2] if reuse_freqs_front_part else [max_pos, 1, 1, d]
//   sin:          [max_pos, 1, 1, d // 2] if reuse_freqs_front_part else [max_pos, 1, 1, d]
//   positions:    [s, b]
//   offsets:      [s, b]
//   rotate_style: 0: NEOX style, 1: GPT-J style
void rope_cached_positions_offsets_fwd_impl(
    torch::Tensor&       output,
    const torch::Tensor& input,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& positions,
    const torch::Tensor& offsets,
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first)
{
    // Get sizes of input and output
    const int32_t size_s = min(min(input.size(0), positions.size(0)), offsets.size(0));
    const int32_t size_b = min(min(input.size(1), positions.size(1)), offsets.size(1));
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

    // Get strides of positions and offsets
    assert(1 == positions.stride(1) && 2 == positions.dim());
    assert(1 == offsets.stride(1)   && 2 == offsets.dim());
    const int32_t max_position = cos.size(0);

    DISPATCH_ROPE_TYPES_PARAMS(
        input.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_1c_sbhd_cached_indirect2<OpCachedFwd, ...>",
        dispatch_1c_sbhd_cached_indirect2<OpCachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst>(
            output.data_ptr<scalar_t_0>(),
            input.data_ptr<scalar_t_0>(),
            cos.data_ptr<scalar_t_1>(),
            sin.data_ptr<scalar_t_1>(),
            positions.data_ptr<int64_t>(),
            offsets.data_ptr<int64_t>(),
            max_position,
            size_s, size_b, size_h, size_d,
            size_f, // size of last dimension of freqs.
            stride_i_s, stride_i_b, stride_i_h, stride_i_d,
            stride_o_s, stride_o_b, stride_o_h, stride_o_d););
}

// Parameters:
//   output_x/y:   [s, b, h, d]
//   input_x/y:    [s, b, h, d]
//   cos:          [max_pos, 1, 1, d // 2] if reuse_freqs_front_part else [max_pos, 1, 1, d]
//   sin:          [max_pos, 1, 1, d // 2] if reuse_freqs_front_part else [max_pos, 1, 1, d]
//   positions:    [s, b]
//   offsets:      [s, b]
//   rotate_style: 0: NEOX style, 1: GPT-J style
void rope_cached_positions_offsets_2c_fwd_impl(
    torch::Tensor&       output_x,
    torch::Tensor&       output_y,
    const torch::Tensor& input_x,
    const torch::Tensor& input_y,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& positions,
    const torch::Tensor& offsets,
    const int32_t        rotate_style,
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
    const int32_t max_position = cos.size(0);

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
            max_position,
            size_s, size_b, size_h_x, size_h_y, size_d,
            size_f, // size of last dimension of freqs.
            stride_ix_s, stride_ix_b, stride_ix_h, stride_ix_d,
            stride_iy_s, stride_iy_b, stride_iy_h, stride_iy_d,
            stride_ox_s, stride_ox_b, stride_ox_h, stride_ox_d,
            stride_oy_s, stride_oy_b, stride_oy_h, stride_oy_d););
}