// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"

void rope_fwd_impl(
    aiter_tensor_t&       output,                    // [s, b, h, d]
    const aiter_tensor_t& input,                     // [s, b, h, d]
    const aiter_tensor_t& freqs,                     // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_bwd_impl(
    aiter_tensor_t&       input_grads,               // [s, b, h, d]
    const aiter_tensor_t& output_grads,              // [s, b, h, d]
    const aiter_tensor_t& freqs,                     // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_2c_fwd_impl(
    aiter_tensor_t&       output_x,                  // [s, b, h, d]
    aiter_tensor_t&       output_y,                  // [s, b, h, d]
    const aiter_tensor_t& input_x,                   // [s, b, h, d]
    const aiter_tensor_t& input_y,                   // [s, b, h, d]
    const aiter_tensor_t& freqs,                     // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_2c_bwd_impl(
    aiter_tensor_t&       input_grads_x,             // [s, b, h, d]
    aiter_tensor_t&       input_grads_y,             // [s, b, h, d]
    const aiter_tensor_t& output_grads_x,            // [s, b, h, d]
    const aiter_tensor_t& output_grads_y,            // [s, b, h, d]
    const aiter_tensor_t& freqs,                     // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_fwd_impl(
    aiter_tensor_t&       output,                    // [s, b, h, d]
    const aiter_tensor_t& input,                     // [s, b, h, d]
    const aiter_tensor_t& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_bwd_impl(
    aiter_tensor_t&       input_grads,               // [s, b, h, d]
    const aiter_tensor_t& output_grads,              // [s, b, h, d]
    const aiter_tensor_t& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_2c_fwd_impl(
    aiter_tensor_t&       output_x,                  // [s, b, h, d]
    aiter_tensor_t&       output_y,                  // [s, b, h, d]
    const aiter_tensor_t& input_x,                   // [s, b, h, d]
    const aiter_tensor_t& input_y,                   // [s, b, h, d]
    const aiter_tensor_t& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_2c_bwd_impl(
    aiter_tensor_t&       input_grads_x,             // [s, b, h, d]
    aiter_tensor_t&       input_grads_y,             // [s, b, h, d]
    const aiter_tensor_t& output_grads_x,            // [s, b, h, d]
    const aiter_tensor_t& output_grads_y,            // [s, b, h, d]
    const aiter_tensor_t& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_positions_fwd_impl(
    aiter_tensor_t&       output,                    // [s, b, h, d]
    const aiter_tensor_t& input,                     // [s, b, h, d]
    const aiter_tensor_t& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& positions,                 // [s, b]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_positions_2c_fwd_impl(
    aiter_tensor_t&       output_x,                  // [s, b, h, d]
    aiter_tensor_t&       output_y,                  // [s, b, h, d]
    const aiter_tensor_t& input_x,                   // [s, b, h, d]
    const aiter_tensor_t& input_y,                   // [s, b, h, d]
    const aiter_tensor_t& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& positions,                 // [s, b]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_positions_offsets_fwd_impl(
    aiter_tensor_t&       output,                    // [s, b, h, d]
    const aiter_tensor_t& input,                     // [s, b, h, d]
    const aiter_tensor_t& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& positions,                 // [s, b]
    const aiter_tensor_t& offsets,                   // [s, b]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_cached_positions_offsets_2c_fwd_impl(
    aiter_tensor_t&       output_x,                  // [s, b, h, d]
    aiter_tensor_t&       output_y,                  // [s, b, h, d]
    const aiter_tensor_t& input_x,                   // [s, b, h, d]
    const aiter_tensor_t& input_y,                   // [s, b, h, d]
    const aiter_tensor_t& cos,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& sin,                       // [s, 1, 1, d // 2] if reuse_freqs_front_part else [s, 1, 1, d]
    const aiter_tensor_t& positions,                 // [s, b]
    const aiter_tensor_t& offsets,                   // [s, b]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_thd_fwd_impl(
    aiter_tensor_t&       output,                    // [t, h, d]
    const aiter_tensor_t& input,                     // [t, h, d]
    const aiter_tensor_t& cu_seqlens,                // [b + 1]
    const aiter_tensor_t& freqs,                     // [max_s, 1, 1, d // 2] if reuse_freqs_front_part else [max_s, 1, 1, d], where max_s = cu_seqlens[-1]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_thd_bwd_impl(
    aiter_tensor_t&       input_grads,               // [t, h, d]
    const aiter_tensor_t& output_grads,              // [t, h, d]
    const aiter_tensor_t& cu_seqlens,                // [b + 1]
    const aiter_tensor_t& freqs,                     // [max_s, 1, 1, d // 2] if reuse_freqs_front_part else [max_s, 1, 1, d], where max_s = cu_seqlens[-1]
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_2d_fwd_impl(
    aiter_tensor_t&       output,                    // [b, s, h, d] where s = H * W
    const aiter_tensor_t& input,                     // [b, s, h, d] where s = H * W
    const aiter_tensor_t& cos_h,                     // [1, H', 1,  d // 4] if reuse_freqs_front_part else [1, H', 1,  d // 2], where H' >= H
    const aiter_tensor_t& sin_h,                     // [1, H', 1,  d // 4] if reuse_freqs_front_part else [1, H', 1,  d // 2], where H' >= H
    const aiter_tensor_t& cos_w,                     // [1, 1,  W', d // 4] if reuse_freqs_front_part else [1, 1,  W', d // 2], where W' >= W
    const aiter_tensor_t& sin_w,                     // [1, 1,  W', d // 4] if reuse_freqs_front_part else [1, 1,  W', d // 2], where W' >= W
    const int32_t        img_height,                // H
    const int32_t        img_width,                 // W
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);

void rope_2d_bwd_impl(
    aiter_tensor_t&       input_grads,               // [b, s, h, d] where s = H * W
    const aiter_tensor_t& output_grads,              // [b, s, h, d] where s = H * W
    const aiter_tensor_t& cos_h,                     // [1, H', 1,  d // 4] if reuse_freqs_front_part else [1, H', 1,  d // 2], where H' >= H
    const aiter_tensor_t& sin_h,                     // [1, H', 1,  d // 4] if reuse_freqs_front_part else [1, H', 1,  d // 2], where H' >= H
    const aiter_tensor_t& cos_w,                     // [1, 1,  W', d // 4] if reuse_freqs_front_part else [1, 1,  W', d // 2], where W' >= W
    const aiter_tensor_t& sin_w,                     // [1, 1,  W', d // 4] if reuse_freqs_front_part else [1, 1,  W', d // 2], where W' >= W
    const int32_t        img_height,                // H
    const int32_t        img_width,                 // W
    const int32_t        rotate_style,              // 0: NEOX style, 1: GPT-J style
    const bool           reuse_freqs_front_part,
    const bool           nope_first
);
