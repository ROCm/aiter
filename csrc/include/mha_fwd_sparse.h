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
    // VFA only: number of ONLINE no-mask KV blocks to run before freezing the
    // softmax running max. <=0 is treated as 1 (freeze after the warm-up
    // block). Ignored by the non-VFA sparse kernels.
    int freeze_softmax_max_count = 1;
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

// VFA ("frozen-max") kernarg blob: the 704-byte sparse blob plus a 16-byte
// tail whose first dword is freeze_softmax_max_count (kernarg offset 0x2C0,
// read into s[73] by mi350_fmha_hd128_i8fp8_sparse_vfa.py). The remaining 12
// bytes are padding to keep the blob 16-byte aligned (kernarg_segment_size=720).
struct __attribute__((packed)) fmha_fwd_v3_sparse_vfa_args : public fmha_fwd_v3_sparse_args
{
    int32_t s_freeze_softmax_max_count;
    int32_t _pad_freeze[3];
};

static_assert(sizeof(fmha_fwd_v3_sparse_vfa_args) == 720,
              "fmha_fwd_v3_sparse_vfa_args must be exactly 720 bytes "
              "(matches the @kernel(_kernarg_raw size=720) in "
              "mi350_fmha_hd128_i8fp8_sparse_vfa.py).");

// Sparse dispatcher. Returns the launch time in ms, -1 on unsupported config.
float fmha_fwd_v3_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

// Sparse mxfp4 sibling. Same kernarg layout, different .co
// (fwd_hd128_mxfp4_sparse.co) generated from
// /workspace/mi350_fmha_hd128_mxfp4_sparse.py. Q/K are fp4-packed
// (caller tensor dtype int8/uint8 with last dim = head_dim/2 = 64
// for hd=128), V is fp8, Q/K scales are E8M0 per-block uint8 bytes
// and V descale is fp32 per output channel -- but on the kernel side
// none of those buffer dtypes are baked into the kernarg, so the
// dispatcher just forwards base pointers and lets the wrapper
// validate shapes (see asm_mha_fwd_sparse.cu::fmha_v3_fwd_mxfp4_sparse).
float fmha_fwd_v3_mxfp4_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

// DENSE (non-sparse) mxfp4 sibling. Same fp4-Q/K * fp8-V * bf16-out contract as
// the mxfp4 sparse path, but the kernel (fwd_hd128_mxfp4.co, from
// mi350_fmha_hd128_mxfp4.py) walks all KV blocks sequentially -- no LUT. It reads
// the standard 656-byte dense kernarg, so the launcher reuses init_sparse_v3_args
// (which fills the dense 656-byte prefix) and copies only those bytes.
float fmha_fwd_v3_mxfp4(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

// Sparse fp8 sibling. Same kernarg layout, different .co
// (fwd_hd128_fp8_sparse.co) generated from
// /workspace/mi350_fmha_hd128_fp8_sparse.py. Q/K/V are all fp8 (E4M3)
// and the descales are per-tensor fp32, so the dispatcher reuses the
// i8fp8 init_sparse_v3_args path verbatim (in_bpe=1 for fp8).
float fmha_fwd_v3_fp8_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

// Sparse i8fp8 VFA ("frozen-max") sibling. Same i8fp8 data contract and
// 704-byte kernarg layout as fmha_fwd_v3_sparse; the only difference is the
// .co (fwd_hd128_i8fp8_sparse_vfa.co) generated from
// mi350_fmha_hd128_i8fp8_sparse_vfa.py, whose no-mask inner blocks use the
// frozen-max softmax (mimics fav3_sage_attention.py's FROZEN_MAX path). So
// init_sparse_v3_args is reused unchanged; only the kernel symbol + .co differ.
float fmha_fwd_v3_i8fp8_sparse_vfa(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

// Sparse fp8 VFA ("frozen-max") sibling. Same fp8 data contract as
// fmha_fwd_v3_fp8_sparse and the same 720-byte VFA kernarg blob as
// fmha_fwd_v3_i8fp8_sparse_vfa; the only difference is the .co
// (fwd_hd128_fp8_sparse_vfa.co) generated from mi350_fmha_hd128_fp8_sparse_vfa.py,
// whose no-mask inner blocks use the frozen-max softmax. init_sparse_v3_args is
// reused unchanged (in_bpe=1 for fp8); only the kernel symbol + .co differ.
float fmha_fwd_v3_fp8_sparse_vfa(mha_fwd_sparse_args a, const ck_tile::stream_config& s);

} // namespace aiter
