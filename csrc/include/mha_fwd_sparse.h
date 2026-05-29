#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Block-sparse FMHA forward (Sage i8fp8, hd=128, gfx950).
// Sibling of mha_fwd.h / fmha_fwd_v3_args, extended with the 3 LUT pointers
// that the hand-written ASM kernel consumes at kernarg offsets 0x290/0x2A0/
// 0x2B0 (see /workspace/mi350_fmha_hd128_i8fp8_sparse.py).

#include "aiter_hip_common.h"
#include "mha_fwd.h"

namespace aiter {

// Block-sparse args. Augments the dense mha_fwd_args (no inheritance to keep
// the existing struct's POD-ness intact) with the 3 LUT tensor pointers
// produced by aiter.ops.triton.attention.utils.block_attn_mask_to_ragged_lut.
struct mha_fwd_sparse_args : public mha_fwd_args
{
    const void* kv_block_indices_ptr; // int32, shape [lut_count.sum()]
    const void* lut_start_ptr;        // int32, shape [B*HQ*num_q_blocks]
    const void* lut_count_ptr;        // int32, same shape
};

// On-device kernarg blob: same 656 bytes as fmha_fwd_v3_args + 48 bytes
// (3 ptr slots with the same p2 padding convention).
struct __attribute__((packed)) fmha_fwd_v3_sparse_args : public fmha_fwd_v3_args
{
    const void* ptr_kv_block_indices;
    p2 _ppad_kv;
    const void* ptr_lut_start;
    p2 _ppad_ls;
    const void* ptr_lut_count;
    p2 _ppad_lc;
};

static_assert(sizeof(fmha_fwd_v3_sparse_args) == 704,
              "fmha_fwd_v3_sparse_args must be exactly 704 bytes "
              "(matches the @kernel(_kernarg_raw size=704) in "
              "mi350_fmha_hd128_i8fp8_sparse.py).");

// Sparse dispatcher. Returns the launch time in ms, -1 on unsupported config.
float fmha_fwd_v3_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

} // namespace aiter
