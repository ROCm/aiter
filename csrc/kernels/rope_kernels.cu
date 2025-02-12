// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "dispatch_utils.h"

void rope_fwd(
    torch::Tensor&       output,        // [s, b, h, d]
    const torch::Tensor& input,         // [s, b, h, d]
    const torch::Tensor& freqs)         // [s, 1, 1, d]
{
    const int s = input.size(0);
}

void rope_bwd(
    torch::Tensor&       input_grads,   // [s, b, h, d]
    const torch::Tensor& output_grads,  // [s, b, h, d]
    const torch::Tensor& freqs)         // [s, 1, 1, d]
{
}

void rope_cached_fwd(
    torch::Tensor&       output,        // [s, b, h, d]
    const torch::Tensor& input,         // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin)           // [s, 1, 1, d]
{

}

void rope_cached_bwd(
    torch::Tensor&       input_grads,   // [s, b, h, d]
    const torch::Tensor& output_grads,  // [s, b, h, d]
    const torch::Tensor& cos,           // [s, 1, 1, d]
    const torch::Tensor& sin)           // [s, 1, 1, d]
{

}

void rope_thd_fwd(
    torch::Tensor&       output,        // [t, h, d]
    const torch::Tensor& input,         // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [t]
    const torch::Tensor& freqs)         // [t, d]
{

}

void rope_thd_bwd(
    torch::Tensor&       input_grads,   // [t, h, d]
    const torch::Tensor& output_grads,  // [t, h, d]
    const torch::Tensor& cu_seqlens,    // [t]
    const torch::Tensor& freqs)         // [t, d]
{

}

void rope_2d_fwd(
    torch::Tensor&       output,
    const torch::Tensor& input,
    const torch::Tensor& cos_height,
    const torch::Tensor& sin_height,
    const torch::Tensor& cos_width,
    const torch::Tensor& sin_width)
{

}

void rope_2d_bwd(
    torch::Tensor&       input_grads,
    const torch::Tensor& output_grads,
    const torch::Tensor& cos_height,
    const torch::Tensor& sin_height,
    const torch::Tensor& cos_width,
    const torch::Tensor& sin_width)
{

}
