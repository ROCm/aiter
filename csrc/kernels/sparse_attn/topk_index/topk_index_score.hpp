// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// __global__ entry, per-variant host launchers and the instantiation table for
// the bf16-Q x fp8-K decode lightning-indexer scoring pass. The device template
// is in topk_index_score_kernels.cuh; the ctypes entry and the grid lever are
// not here. The incumbent fp8-Q path (pa_sparse_block_select.hpp) is neither
// included nor modified.

#include "aiter_hip_common.h"
#include "topk_index_score_kernels.cuh"

// Build constants, fixed like the incumbent's (SPARSE_HEAD_DIM /
// SPARSE_BLOCK_SIZE). head_dim = 128 is not a default: invariant I5's 1.00 L1
// accesses per 128 B line is derived for 128-byte rows.
#define OPUS_IDX_HEAD_DIM 128
#define OPUS_IDX_BLOCK_SIZE 128

// Launcher signature. It carries sm_scale (ledger D1: the frozen contract takes
// it and applies sm_scale*log2e in-kernel) and whole-tensor element counts,
// which become the buffer-descriptor extents -- that is what buys the hardware
// bounds check the page offset relies on (invariant I3).
#define OPUS_IDX_SCORE_PARAMS                                                                 \
    const void *q_idx, const void *key_cache_idx, float *score, const int *block_table,       \
        const int *seq_lens, long long q_numel, long long key_cache_numel,                    \
        long long score_numel, long long block_table_numel, int batch, int num_chunks,        \
        int chunk_blocks, long long stride_q_n, long long stride_q_h, long long stride_ik_blk, \
        long long stride_s_h, long long stride_s_b, long long stride_bt_b, float sm_scale,    \
        hipStream_t stream

#define OPUS_IDX_SCORE_ARGS                                                                    \
    q_idx, key_cache_idx, score, block_table, seq_lens, q_numel, key_cache_numel, score_numel, \
        block_table_numel, batch, num_chunks, chunk_blocks, stride_q_n, stride_q_h,            \
        stride_ik_blk, stride_s_h, stride_s_b, stride_bt_b, sm_scale, stream

namespace aiter {
namespace sparse_attn {

// __launch_bounds__(256, 2) is part of the register-allocation context the
// accepted build was measured in, not a decoration: the occupancy premise the
// grid lever rests on depends on it.
template <int H, int Q, int AUX_K>
__global__ __launch_bounds__(kOpusNumWarps* kOpusWarpSize, 2) void opus_decode_index_score_kernel(
    const opus_decode_score_args a)
{
    opus_index_score::decode_index_score_impl<
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
        opus::fp8_t,
#else
        uint8_t, // stub pass: 1 byte, so the sizeof(K_T) == 1 refusal still holds
#endif
        OPUS_IDX_HEAD_DIM,
        OPUS_IDX_BLOCK_SIZE,
        H,
        Q,
        AUX_K>(a);
}

// Grid is (batch, num_chunks). num_chunks and chunk_blocks come from the
// caller's capture-time constants; this launcher reads no device tensor and
// allocates nothing, so the grid is fixed at capture (cudagraph-safe).
template <int H, int Q, int AUX_K>
void launch_opus_decode_index_score(OPUS_IDX_SCORE_PARAMS)
{
    // G4: no zero-extent grid may be captured. The entry point also rejects
    // num_reqs == 0 earlier; this is the defense-in-depth layer, matching the
    // incumbent's convention of re-checking across layers.
    if(batch <= 0 || num_chunks <= 0)
        return;

    opus_decode_score_args a{};
    a.q_ptr         = q_idx;
    a.ik_ptr        = key_cache_idx;
    a.score_ptr     = score;
    a.bt_ptr        = block_table;
    a.seq_lens_ptr  = seq_lens;
    a.q_numel       = q_numel;
    a.ik_numel      = key_cache_numel;
    a.score_numel   = score_numel;
    a.bt_numel      = block_table_numel;
    a.batch         = batch;
    a.chunk_blocks  = chunk_blocks;
    a.stride_q_n    = stride_q_n;
    a.stride_q_h    = stride_q_h;
    a.stride_ik_blk = stride_ik_blk;
    a.stride_s_h    = stride_s_h;
    a.stride_s_b    = stride_s_b;
    a.stride_bt_b   = stride_bt_b;
    a.sm_scale      = sm_scale;

    dim3 grid((unsigned)batch, (unsigned)num_chunks, 1);
    dim3 block((unsigned)(kOpusNumWarps * kOpusWarpSize), 1, 1);

    opus_decode_index_score_kernel<H, Q, AUX_K><<<grid, block, 0, stream>>>(a);
}

// Host-side wave geometry, so the entry point and any future host-side sizing
// read the same numbers the kernel was built with instead of restating them.
constexpr int kOpusIdxScoreThreads = kOpusNumWarps * kOpusWarpSize; // 256

// ---------------------------------------------------------------------------
// Certification data (NOT dispatch)
// ---------------------------------------------------------------------------
// True only for cells with per-cell bitwise evidence; the entry turns a false
// here into a refusal. Kept separate from the built set even while the two
// agree: they diverge as soon as a cell is built ahead of its evidence.
constexpr bool opus_idx_score_cell_certified(int H, int Q) {
  return (H == 1 && (Q == 1 || Q == 2 || Q == 4 || Q == 8))
         || (H == 4 && (Q == 1 || Q == 2 || Q == 4));
}

} // namespace sparse_attn
} // namespace aiter

// ---------------------------------------------------------------------------
// Per-variant launchers
// ---------------------------------------------------------------------------
#define OPUS_IDX_SCORE_FN(H, Q, A) opus_idx_score_h##H##_q##Q##_aux##A

#define OPUS_IDX_SCORE_DECLARE(H, Q, A) void OPUS_IDX_SCORE_FN(H, Q, A)(OPUS_IDX_SCORE_PARAMS);

#define OPUS_IDX_SCORE_DEFINE(H, Q, A)                                  \
    void OPUS_IDX_SCORE_FN(H, Q, A)(OPUS_IDX_SCORE_PARAMS)              \
    {                                                                   \
        launch_opus_decode_index_score<H, Q, A>(OPUS_IDX_SCORE_ARGS);    \
    }

// ---------------------------------------------------------------------------
// Instantiation table -- (index heads, query tokens, index-K cache policy)
// ---------------------------------------------------------------------------
// (H, Q) is shape-determined by the call; AUX_K in {0, 3} is the tuned axis.
// (4, 8) exceeds the 16 MFMA columns; H = 2 is a deliberate refusal.
#define OPUS_IDX_SCORE_TABLE(F)                                        \
    F(1, 1, 0) F(1, 1, 3) F(1, 2, 0) F(1, 2, 3) F(1, 4, 0) F(1, 4, 3)  \
    F(1, 8, 0) F(1, 8, 3) F(4, 1, 0) F(4, 1, 3) F(4, 2, 0) F(4, 2, 3)  \
    F(4, 4, 0) F(4, 4, 3)

namespace aiter {
namespace sparse_attn {
OPUS_IDX_SCORE_TABLE(OPUS_IDX_SCORE_DECLARE)
} // namespace sparse_attn
} // namespace aiter
