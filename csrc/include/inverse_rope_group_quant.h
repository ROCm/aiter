// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

// DeepSeek-V4 output path helper:
//   input  o       [S, H, head_dim] bf16/fp16, before inverse RoPE
//   output x_fp8   [S, G, D] fp8, where D = H*head_dim/G
//   output x_scale e8m0 scale bytes
// Applies GPT-J inverse RoPE to every head's rope tail, then group-quantizes the
// flattened per-group rows for the upcoming wo_a grouped FP8 BMM.
//
// scale_shuffle:
//   false = row-major [S, G, Ks], Ks = D / quant_group_size
//   true  = MFMA tile-shuffled for V_MFMA_SCALE_F32_16x16x128_F8 (gfx950)
//       Storage [G, S_pad, Ks_pad] flat with 256-byte tiles of [32_M, 8_K].
//       Tile-internal: byte = lane*4 + iter, lane = (k%4)*16 + (m%16),
//       iter = ((m/16)&1) + ((k/4)&1)*2.  S_pad = ceil(S,32), Ks_pad = ceil(Ks,8).
void inverse_rope_group_quant(
    aiter_tensor_t& o,
    aiter_tensor_t& x_fp8,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& positions,
    aiter_tensor_t& cos_cache,
    aiter_tensor_t& sin_cache,
    int64_t num_groups,
    int64_t quant_group_size = 128,
    bool scale_shuffle       = false);

} // namespace aiter
