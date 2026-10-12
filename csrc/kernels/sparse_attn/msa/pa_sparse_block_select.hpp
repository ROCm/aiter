// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "pa_sparse_block_select_kernels.cuh"

// constants shared by all variants
#define SPARSE_HEAD_DIM 128
#define SPARSE_BLOCK_SIZE 128
#define SPARSE_TOPK 16
#define SPARSE_PAGES_PER_BLOCK 8
#define SPARSE_DECODE_QTILES 1
#define SPARSE_DECODE_PREFETCH 8
#define SPARSE_DECODE_NONTEMPORAL false
#define SPARSE_PREFILL_WAVES 4

#define SPARSE_SCORE_DECODE_PARAMS                                                            \
    const uint8_t *q_idx, const uint8_t *key_cache_idx, float *score, const int *block_table, \
        const int *seq_lens, int num_reqs, int num_q_tiles, int block_table_stride,           \
        int score_head_stride, int score_num_stride, int num_chunks, int init_blocks,         \
        int local_blocks, int query_len, hipStream_t stream
#define SPARSE_SCORE_DECODE_ARGS                                                                   \
    q_idx, key_cache_idx, score, block_table, seq_lens, num_reqs, num_q_tiles, block_table_stride, \
        score_head_stride, score_num_stride, num_chunks, init_blocks, local_blocks, query_len,     \
        stream

// chunk_blocks is the pages one chunk covers and num_chunks is how many of them
// the longest reach needs, so the two multiply out to that reach.
#define SPARSE_SCORE_PREFILL_PARAMS                                                           \
    const uint8_t *q_idx, const uint8_t *key_cache_idx, float *score, const int *block_table, \
        const int *cu_seqlens_q, const int *seq_lens, int num_reqs, int num_q_groups,         \
        int block_table_stride, int score_head_stride, int score_num_stride, int num_chunks,  \
        int chunk_blocks, int init_blocks, int local_blocks, hipStream_t stream
#define SPARSE_SCORE_PREFILL_ARGS                                                             \
    q_idx, key_cache_idx, score, block_table, cu_seqlens_q, seq_lens, num_reqs, num_q_groups, \
        block_table_stride, score_head_stride, score_num_stride, num_chunks, chunk_blocks,    \
        init_blocks, local_blocks, stream

#define SPARSE_TOPK_PARAMS                                                                       \
    const float *score, int *topk_idx, const int *seq_lens, const int *num_valid_pages,          \
        const int *block_table, const int *row_req_id, const int *kv_lens, int *sparse_bt,       \
        int *sparse_ctx, int num_idx_heads, int total_q, int score_head_stride,                  \
        int score_num_stride, int topk_head_stride, int topk_num_stride, int block_table_stride, \
        int sparse_bt_stride, int num_kv_heads, int query_len, hipStream_t stream
#define SPARSE_TOPK_ARGS                                                                           \
    score, topk_idx, seq_lens, num_valid_pages, block_table, row_req_id, kv_lens, sparse_bt,       \
        sparse_ctx, num_idx_heads, total_q, score_head_stride, score_num_stride, topk_head_stride, \
        topk_num_stride, block_table_stride, sparse_bt_stride, num_kv_heads, query_len, stream

namespace aiter {
namespace sparse_attn {

// ---------------------------------------------------------------------------
// Launch scoring & topk kernels.
// ---------------------------------------------------------------------------

template <int H, int Waves>
void launch_score_decode(SPARSE_SCORE_DECODE_PARAMS)
{
    // Query-tile groups on x, requests on y, the block-range split on z.
    dim3 grid(num_q_tiles, num_reqs, num_chunks);
    dim3 block(Waves * kWave);

    pa_sparse_block_score_decode<SPARSE_BLOCK_SIZE,
                                 SPARSE_HEAD_DIM,
                                 H,
                                 Waves,
                                 SPARSE_DECODE_QTILES,
                                 SPARSE_DECODE_PREFETCH,
                                 SPARSE_DECODE_NONTEMPORAL>
        <<<grid, block, 0, stream>>>(q_idx,
                                     key_cache_idx,
                                     score,
                                     block_table,
                                     seq_lens,
                                     block_table_stride,
                                     score_head_stride,
                                     score_num_stride,
                                     num_chunks,
                                     init_blocks,
                                     local_blocks,
                                     query_len);
}

template <int H, int QTiles>
void launch_score_prefill(SPARSE_SCORE_PREFILL_PARAMS)
{
    // Query-tile groups on x, requests on y, the block-range split on z.
    dim3 grid(num_q_groups, num_reqs, num_chunks);
    dim3 block(SPARSE_PREFILL_WAVES * kWave);

    pa_sparse_block_score_prefill<SPARSE_BLOCK_SIZE,
                                  SPARSE_HEAD_DIM,
                                  H,
                                  SPARSE_PREFILL_WAVES,
                                  QTiles><<<grid, block, 0, stream>>>(q_idx,
                                                                      key_cache_idx,
                                                                      score,
                                                                      block_table,
                                                                      cu_seqlens_q,
                                                                      seq_lens,
                                                                      block_table_stride,
                                                                      score_head_stride,
                                                                      score_num_stride,
                                                                      chunk_blocks,
                                                                      init_blocks,
                                                                      local_blocks);
}

template <int Slots, int Waves>
void launch_topk(SPARSE_TOPK_PARAMS)
{
    dim3 grid(total_q, num_idx_heads);
    dim3 block(Waves * kWave);

    pa_sparse_block_topk_kernel<SPARSE_BLOCK_SIZE,
                                Slots,
                                SPARSE_TOPK,
                                Waves,
                                SPARSE_PAGES_PER_BLOCK>
        <<<grid, block, 0, stream>>>(score,
                                     topk_idx,
                                     seq_lens,
                                     num_valid_pages,
                                     block_table,
                                     row_req_id,
                                     kv_lens,
                                     sparse_bt,
                                     sparse_ctx,
                                     score_head_stride,
                                     score_num_stride,
                                     topk_head_stride,
                                     topk_num_stride,
                                     block_table_stride,
                                     sparse_bt_stride,
                                     num_kv_heads,
                                     query_len);
}

} // namespace sparse_attn
} // namespace aiter

// ---------------------------------------------------------------------------
// Per-variant launchers
// ---------------------------------------------------------------------------
#define SPARSE_DECODE_FN(H, W) sparse_score_decode_h##H##_w##W
#define SPARSE_PREFILL_FN(H, Q) sparse_score_prefill_h##H##_q##Q
#define SPARSE_TOPK_FN(S, W) sparse_topk_s##S##_w##W

#define SPARSE_DECODE_DECLARE(H, W) void SPARSE_DECODE_FN(H, W)(SPARSE_SCORE_DECODE_PARAMS);
#define SPARSE_PREFILL_DECLARE(H, Q) void SPARSE_PREFILL_FN(H, Q)(SPARSE_SCORE_PREFILL_PARAMS);
#define SPARSE_TOPK_DECLARE(S, W) void SPARSE_TOPK_FN(S, W)(SPARSE_TOPK_PARAMS);

#define SPARSE_DECODE_DEFINE(H, W)                           \
    void SPARSE_DECODE_FN(H, W)(SPARSE_SCORE_DECODE_PARAMS)  \
    {                                                        \
        launch_score_decode<H, W>(SPARSE_SCORE_DECODE_ARGS); \
    }
#define SPARSE_PREFILL_DEFINE(H, Q)                            \
    void SPARSE_PREFILL_FN(H, Q)(SPARSE_SCORE_PREFILL_PARAMS)  \
    {                                                          \
        launch_score_prefill<H, Q>(SPARSE_SCORE_PREFILL_ARGS); \
    }
#define SPARSE_TOPK_DEFINE(S, W) \
    void SPARSE_TOPK_FN(S, W)(SPARSE_TOPK_PARAMS) { launch_topk<S, W>(SPARSE_TOPK_ARGS); }

// ---------------------------------------------------------------------------
// scoring & topk kernel instantiation configuration tables
// ---------------------------------------------------------------------------
// (index heads, waves)
#define SPARSE_DECODE_TABLE(F) F(1, 1) F(1, 2) F(1, 4) F(2, 1) F(2, 2) F(2, 4)

// (index heads, query tiles)
#define SPARSE_PREFILL_TABLE(F) F(1, 1) F(1, 2) F(1, 4) F(2, 1) F(2, 2) F(2, 4)

// (slots, waves)
// clang-format off
#define SPARSE_TOPK_TABLE(F)                                                   \
    F(1, 1)                                                                    \
    F(2, 1) F(2, 2)                                                            \
    F(4, 1) F(4, 2) F(4, 4)                                                    \
    F(8, 1) F(8, 2) F(8, 4) F(8, 8)                                            \
    F(16, 1) F(16, 2) F(16, 4) F(16, 8)                                        \
    F(32, 1) F(32, 2) F(32, 4) F(32, 8)                                        \
    F(64, 1) F(64, 2) F(64, 4) F(64, 8)                                        \
    F(128, 1) F(128, 2) F(128, 4) F(128, 8)
// clang-format on

namespace aiter {
namespace sparse_attn {
SPARSE_DECODE_TABLE(SPARSE_DECODE_DECLARE)
SPARSE_PREFILL_TABLE(SPARSE_PREFILL_DECLARE)
SPARSE_TOPK_TABLE(SPARSE_TOPK_DECLARE)
} // namespace sparse_attn
} // namespace aiter
