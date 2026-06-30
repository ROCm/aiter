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

    // ---- Additive persistent/sorted/affine extension (prevfa surgical port). ----
    // Only consumed by the fp8 sparse persistent/sorted/affine_sorted/affine
    // dispatchers below; all default to nullptr/0 so every existing caller and the
    // deployed 704-byte base/VFA paths are byte-for-byte unaffected.
    const void* lut_freeze_ptr = nullptr; // VSA freeze LUT (int32, same shape as lut_count); nullptr => disabled
    const void* work_table_ptr = nullptr; // int32[total_tiles], packed q|(h<<16)|(b<<24), LPT-sorted
    uint32_t    num_wgs        = 0;       // persistent grid size (0 => dispatcher picks the CU count)
    uint32_t    total_tiles    = 0;       // = batch*nhead_q*num_q_blocks (== work_table.numel())
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

// Persistent/sorted/affine kernarg blob (prevfa surgical port). Extends the
// 704-byte sparse layout with a lut_freeze slot (0x2C0), then the work-table
// pointer (0x2D0) and grid-stride/loop scalars (0x2E0/0x2E4), padded to 752 to
// match the @kernel(_kernarg_raw size=752) emitted with _PERSISTENT/_SORTED/
// affine codegen. Byte-identical to the fork's fmha_fwd_v3_sparse_persistent_args
// (whose 720-byte base already carried ptr_lut_freeze): the 3 LUT pointers stay
// at 0x290/0x2A0/0x2B0 and work_table lands at 0x2D0. The deployed 704-byte base
// .co's never see these trailing bytes; they are only sent on the persistent/
// affine dispatchers below.
struct __attribute__((packed)) fmha_fwd_v3_sparse_persistent_args
    : public fmha_fwd_v3_sparse_args
{
    const void* ptr_lut_freeze; // 0x2C0 (VSA; nullptr/ignored by affine_sorted)
    p2 _ppad_lf;
    const void* ptr_work_table; // 0x2D0 (int32[total_tiles])
    p2 _ppad_wt;
    uint32_t s_num_wgs;         // 0x2E0 grid-stride
    uint32_t s_total_tiles;     // 0x2E4 loop bound
    uint64_t _tail_pad;         // pad 744 -> 752 (matches the .co kernarg segment)
};

static_assert(sizeof(fmha_fwd_v3_sparse_persistent_args) == 752,
              "fmha_fwd_v3_sparse_persistent_args must be exactly 752 bytes "
              "(704 sparse + lut_freeze(16) + work_table(16) + num_wgs(4) + "
              "total_tiles(4) + pad(8)); work_table at kernarg 0x2D0.");

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

// ---- Persistent/sorted/affine fp8 sparse dispatchers (prevfa surgical port). ----
// All share the fp8 data contract and the 752-byte persistent kernarg layout
// (the 3 LUT pointers stay at 0x290/0x2A0/0x2B0; work_table at 0x2D0). They are
// additive: the existing fmha_fwd_v3_fp8_sparse / _vfa dispatchers and the
// deployed 704/720-byte .co's are untouched.

// Work-table fp8 sparse path. Requires a.work_table_ptr / a.total_tiles.
//   * sorted_dispatch=true : one WG per tile, flat grid (total_tiles,1,1); each
//     WG reads work_table[wg_id]. Routes to fwd_hd128_fp8_sparse_sorted.co.
//   * sorted_dispatch=false: persistent grid-stride; 1-D grid of a.num_wgs WGs
//     (auto-sized to the CU count when 0). Routes to fwd_hd128_fp8_sparse_persistent.co.
// (These two .co's are NOT shipped in this branch yet; the dispatcher is provided
// so the shared torch entry links. Only affine_sorted below is wired to a shipped .co.)
float fmha_fwd_v3_fp8_sparse_persistent(mha_fwd_sparse_args a,
                                        const ck_tile::stream_config& s,
                                        bool sorted_dispatch = false);

// Sorted-dispatch fp8 sparse variant built from the affine codegen
// (mi350_fmha_hd128_fp8_affine.py). Flat grid (total_tiles,1,1) + a.work_table_ptr,
// routes to fwd_hd128_fp8_sparse_affine_sorted.co (SHIPPED), which reads its LUT
// pointers from the dead group-mode arg slots (0x1B0/0x1C0/0x1E0) rather than the
// 0x290 tail. Uses the BASE fp8 sparse kernel symbol
// (_ZN5aiter32fmha_fwd_hd128_fp8_sparse_gfx950E). Requires a.work_table_ptr/a.total_tiles.
float fmha_fwd_v3_fp8_sparse_affine_sorted(mha_fwd_sparse_args a,
                                           const ck_tile::stream_config& s);

// Non-sorted (XCD-swizzle) affine fp8 sparse variant. Same affine group-mode-slot
// LUT ABI as affine_sorted, but the standard (num_q_blocks, nhead_q, batch) grid +
// 656-byte dense kernarg (no work_table). Routes to fwd_hd128_fp8_sparse_affine.co
// (NOT shipped in this branch yet); provided so the shared torch entry links.
float fmha_fwd_v3_fp8_sparse_affine(mha_fwd_sparse_args a,
                                    const ck_tile::stream_config& s);

} // namespace aiter
