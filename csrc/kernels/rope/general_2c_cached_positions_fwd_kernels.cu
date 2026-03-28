// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rope_common.h"
using namespace aiter;

/**
 * @brief Compute Rotational Positional Encoding on 2 channels: @param input_x and @param input_y. Results are written
 *        in @param output_x and @param output_y respectively.
 *        Cosine and sine of frequency should have been calculated and specified in @param cos and @param sin.
 *        @param positions is an indirect buffer storing the index of value in @param cos and @param sin used to
 *        calculate with current input element.
 *
 * @param output_x     [s, b, h, d]
 * @param output_y     [s, b, h, d]
 * @param input_x      [s, b, h, d]
 * @param input_y      [s, b, h, d]
 * @param cos          [max_pos, 1, 1, d // 2] if @param reuse_freqs_front_part else [max_pos, 1, 1, d]
 * @param sin          [max_pos, 1, 1, d // 2] if @param reuse_freqs_front_part else [max_pos, 1, 1, d]
 * @param positions    [s, b]
 * @param rotate_style 0: NEOX style, 1: GPT-J style
 * @param nope_first   If true, back part in last dimension of input is rotated. Otherwise, the front part is rotated.
 */
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

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input_x));
    DISPATCH_ROPE_TYPES_PARAMS(
        input_x.scalar_type(),
        cos.scalar_type(),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        "dispatch_2c_sbhd_cached_indirect<OpCachedFwd, ...>",
        dispatch_2c_sbhd_cached_indirect<OpCachedFwd, RotateStyle, ReuseFreqsFrontPart, NopeFirst, true>(
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
