// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

// DeepSeek-V4 output path helper:
//   input  o       [S, H, head_dim] bf16/fp16, before inverse RoPE
//   output x_fp8   [S, G, D] fp8, where D = H*head_dim/G
//   output x_scale [S, G, ceil(D/group)] e8m0, row- or column-major
// Applies GPT-J inverse RoPE to every head's rope tail, then group-quantizes the
// flattened per-group rows for the upcoming wo_a grouped FP8 BMM.
//
// `transpose_scale` selects the column-major scale storage that the
// preshuffled-B blockscale GEMM reads (ATOM's per_1x128 `transpose_scale`). It is
// a plain transpose, NOT the MX `mx_scale_shuffle_idx` swizzle -- that layout
// belongs to the group-32 MX/MoE GEMMs. Only the caller's x_scale strides
// change; the logical shape is [S, G, ceil(D/group)] either way.
void inverse_rope_group_quant(
    aiter_tensor_t& o,
    aiter_tensor_t& x_fp8,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& positions,
    aiter_tensor_t& cos_cache,
    aiter_tensor_t& sin_cache,
    int64_t num_groups,
    int64_t quant_group_size = 128,
    bool transpose_scale     = false);

} // namespace aiter
