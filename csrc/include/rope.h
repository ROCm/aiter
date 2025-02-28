// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

void rope_fwd_impl(
    torch::Tensor&       output,                    // [s, b, h, d]
    const torch::Tensor& input,                     // [s, b, h, d]
    const torch::Tensor& freqs,                     // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_bwd_impl(
    torch::Tensor&       input_grads,               // [s, b, h, d]
    const torch::Tensor& output_grads,              // [s, b, h, d]
    const torch::Tensor& freqs,                     // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_2c_fwd_impl(
    torch::Tensor&       output_x,                  // [s, b, h, d]
    torch::Tensor&       output_y,                  // [s, b, h, d]
    const torch::Tensor& input_x,                   // [s, b, h, d]
    const torch::Tensor& input_y,                   // [s, b, h, d]
    const torch::Tensor& freqs,                     // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_2c_bwd_impl(
    torch::Tensor&       input_grads_x,             // [s, b, h, d]
    torch::Tensor&       input_grads_y,             // [s, b, h, d]
    const torch::Tensor& output_grads_x,            // [s, b, h, d]
    const torch::Tensor& output_grads_y,            // [s, b, h, d]
    const torch::Tensor& freqs,                     // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_fwd_impl(
    torch::Tensor&       output,                    // [s, b, h, d]
    const torch::Tensor& input,                     // [s, b, h, d]
    const torch::Tensor& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_bwd_impl(
    torch::Tensor&       input_grads,               // [s, b, h, d]
    const torch::Tensor& output_grads,              // [s, b, h, d]
    const torch::Tensor& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_2c_fwd_impl(
    torch::Tensor&       output_x,                  // [s, b, h, d]
    torch::Tensor&       output_y,                  // [s, b, h, d]
    const torch::Tensor& input_x,                   // [s, b, h, d]
    const torch::Tensor& input_y,                   // [s, b, h, d]
    const torch::Tensor& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_2c_bwd_impl(
    torch::Tensor&       input_grads_x,             // [s, b, h, d]
    torch::Tensor&       input_grads_y,             // [s, b, h, d]
    const torch::Tensor& output_grads_x,            // [s, b, h, d]
    const torch::Tensor& output_grads_y,            // [s, b, h, d]
    const torch::Tensor& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_positions_2c_fwd_impl(
    torch::Tensor&       output_x,                  // [s, b, h, d]
    torch::Tensor&       output_y,                  // [s, b, h, d]
    const torch::Tensor& input_x,                   // [s, b, h, d]
    const torch::Tensor& input_y,                   // [s, b, h, d]
    const torch::Tensor& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& positions,                 // [s]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_positions_offsets_2c_fwd_impl(
    torch::Tensor&       output_x,                  // [s, b, h, d]
    torch::Tensor&       output_y,                  // [s, b, h, d]
    const torch::Tensor& input_x,                   // [s, b, h, d]
    const torch::Tensor& input_y,                   // [s, b, h, d]
    const torch::Tensor& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const torch::Tensor& positions,                 // [s]
    const torch::Tensor& offsets,                   // [s]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_thd_fwd_impl(
    torch::Tensor&       output,                    // [t, h, d]
    const torch::Tensor& input,                     // [t, h, d]
    const torch::Tensor& cu_seqlens,                // [b + 1]
    const torch::Tensor& freqs,                     // [max_s, 1, 1, d // 2] if reuse_freqs_front_part else [max_s, 1, 1, d], where max_s = cu_seqlens[-1]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_thd_bwd_impl(
    torch::Tensor&       input_grads,               // [t, h, d]
    const torch::Tensor& output_grads,              // [t, h, d]
    const torch::Tensor& cu_seqlens,                // [b + 1]
    const torch::Tensor& freqs,                     // [max_s, 1, 1, d // 2] if reuse_freqs_front_part else [max_s, 1, 1, d], where max_s = cu_seqlens[-1]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_2d_fwd_impl(
    torch::Tensor&       output,                    // [b, s, h, d] where s = H * W
    const torch::Tensor& input,                     // [b, s, h, d] where s = H * W
    const torch::Tensor& cos_h,                     // [1, H', 1,  d // 4] if reuse_freqs_front_part else [1, H', 1,  d // 2], where H' >= H
    const torch::Tensor& sin_h,                     // [1, H', 1,  d // 4] if reuse_freqs_front_part else [1, H', 1,  d // 2], where H' >= H
    const torch::Tensor& cos_w,                     // [1, 1,  W', d // 4] if reuse_freqs_front_part else [1, 1,  W', d // 2], where W' >= W
    const torch::Tensor& sin_w,                     // [1, 1,  W', d // 4] if reuse_freqs_front_part else [1, 1,  W', d // 2], where W' >= W
    const int32_t        img_height,                // H
    const int32_t        img_width,                 // W
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_2d_bwd_impl(
    torch::Tensor&       input_grads,               // [b, s, h, d] where s = H * W
    const torch::Tensor& output_grads,              // [b, s, h, d] where s = H * W
    const torch::Tensor& cos_h,                     // [1, H', 1,  d // 4] if reuse_freqs_front_part else [1, H', 1,  d // 2], where H' >= H
    const torch::Tensor& sin_h,                     // [1, H', 1,  d // 4] if reuse_freqs_front_part else [1, H', 1,  d // 2], where H' >= H
    const torch::Tensor& cos_w,                     // [1, 1,  W', d // 4] if reuse_freqs_front_part else [1, 1,  W', d // 2], where W' >= W
    const torch::Tensor& sin_w,                     // [1, 1,  W', d // 4] if reuse_freqs_front_part else [1, 1,  W', d // 2], where W' >= W
    const int32_t        img_height,                // H
    const int32_t        img_width,                 // W
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);
