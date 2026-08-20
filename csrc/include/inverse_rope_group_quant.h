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
// scale_layout picks how the e8m0 bytes are laid out for the consuming GEMM.
// Ks = D / quant_group_size throughout.
enum ScaleLayout : int64_t
{
    // Row-major [S, G, Ks], unit stride on Ks.
    kScaleRowMajor = 0,
    // MFMA tile-shuffled for V_MFMA_SCALE_F32_16x16x128_F8 (gfx950).
    //   Storage [G, S_pad, Ks_pad] flat with 256-byte tiles of [32_M, 8_K].
    //   Tile-internal: byte = lane*4 + iter, lane = (k%4)*16 + (m%16),
    //   iter = ((m/16)&1) + ((k/4)&1)*2.  S_pad = ceil(S,32), Ks_pad = ceil(Ks,8).
    //   Same byte permutation as mx_scale_shuffle_idx, but k means a quant group
    //   here rather than a 32-element MX block, so the implementations stay separate.
    kScaleMfmaTile = 1,
    // n32k4, the gfx1250 WMMA scale layout that aiter.ops.shuffle's
    // shuffle_scale_n32k4 produces for weights -- emitted here for the
    // activation so no separate transpose pass is needed.
    //   Storage [ceil(S,32)/32, G, Ks*32], byte =
    //     ((s/32)*G + g)*Ks*32 + (k/4)*128 + (s%32)*4 + (k%4).
    //   The 32 in "n32" is the 32 rows of a super-row, not the quant group.
    //   Requires quant_group_size == 32 and Ks % 4 == 0: the consumer reads a
    //   lane's whole WMMA scaleB operand -- 4 e8m0 of one K=128 step -- with one
    //   ds_load_b32, so the groups come in fours and each must span 128/4 = 32
    //   elements. shuffle_scale_n32k4 pins both the same way (it rejects rather
    //   than pads, and its input shape is (E, N, K//32)).
    kScaleN32K4 = 2,
};

void inverse_rope_group_quant(
    aiter_tensor_t& o,
    aiter_tensor_t& x_fp8,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& positions,
    aiter_tensor_t& cos_cache,
    aiter_tensor_t& sin_cache,
    int64_t num_groups,
    int64_t quant_group_size = 128,
    int64_t scale_layout     = kScaleRowMajor);

} // namespace aiter
