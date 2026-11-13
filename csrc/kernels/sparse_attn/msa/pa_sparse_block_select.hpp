// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "pa_sparse_block_select_kernels.cuh"

// constants shared by all variants
#define SPARSE_HEAD_DIM 128
#define SPARSE_BLOCK_SIZE 128
#define SPARSE_TOPK 16
#define SPARSE_PAGES_PER_BLOCK 8
#define SPARSE_DECODE_PREFETCH 8
#define SPARSE_DECODE_NONTEMPORAL true // nvfp4 only
#define SPARSE_PREFILL_WAVES 4

// rank/world are the round-robin block shard the pass scores, global = local *
// world + rank. world == 1 scores every block, which is the non-partitioned
// case and is bit-identical to having no shard at all.
#define SPARSE_SCORE_DECODE_PARAMS                                                               \
    const uint8_t *q_idx, const uint8_t *key_cache_idx, float *score, const int *block_table,    \
        const int *seq_lens, int num_reqs, int num_q_tiles, int block_table_stride,              \
        int score_head_stride, int score_num_stride, int num_chunks, int init_blocks,            \
        int local_blocks, int query_len, int rank, int world, int page_stride, int req_xor_mask, \
        int fill_shift, hipStream_t stream
#define SPARSE_SCORE_DECODE_ARGS                                                                   \
    q_idx, key_cache_idx, score, block_table, seq_lens, num_reqs, num_q_tiles, block_table_stride, \
        score_head_stride, score_num_stride, num_chunks, init_blocks, local_blocks, query_len,     \
        rank, world, page_stride, req_xor_mask, fill_shift, stream

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

// The context-parallel selector: one launch that nominates this rank's shard,
// publishes it over IPC and merges what the peers sent. Takes the shard's
// scores and the peer mapping, and emits the same page table in the same
// numbering ``SPARSE_TOPK`` above does, so a mixed batch can share one buffer.
#define SPARSE_TOPK_CP_PARAMS                                                                   \
    const float *score, CpPeers peers, uint32_t *cp_gen, int *topk_idx, const int *seq_lens,     \
        const int *block_table, int *sparse_bt, int *sparse_ctx, int num_idx_heads, int total_q, \
        int cand_rows, int score_head_stride, int score_num_stride, int topk_head_stride,        \
        int topk_num_stride, int block_table_stride, int sparse_bt_stride, int num_kv_heads,     \
        int query_len, int init_blocks, int local_blocks, int rank, int world, hipStream_t stream
#define SPARSE_TOPK_CP_ARGS                                                                   \
    score, peers, cp_gen, topk_idx, seq_lens, block_table, sparse_bt, sparse_ctx,              \
        num_idx_heads, total_q, cand_rows, score_head_stride, score_num_stride,                \
        topk_head_stride, topk_num_stride, block_table_stride, sparse_bt_stride, num_kv_heads, \
        query_len, init_blocks, local_blocks, rank, world, stream

// Lanes' slots the merge half needs. Held at one -- a world of 4 sends 64
// candidates and a wave has 64 lanes -- because the merge rides the shard
// pass's launch rather than owning one, so widening it would spend registers
// on every block of a phase that is not the one under pressure.
// The host refuses a world the single slot cannot cover.
#define SPARSE_TOPK_CP_MERGE_SLOTS 1

namespace aiter {
namespace sparse_attn {

// ---------------------------------------------------------------------------
// Launch scoring & topk kernels.
// ---------------------------------------------------------------------------

// num_q_tiles counts groups of QTiles query tiles.
template <int KvFmt, int H, int QTiles, int Waves>
void launch_score_decode(SPARSE_SCORE_DECODE_PARAMS)
{
    // Query-tile groups on x, requests on y, the block-range split on z.
    dim3 grid(num_q_tiles, num_reqs, num_chunks);
    dim3 block(Waves * kWave);

    pa_sparse_block_score_decode<SPARSE_BLOCK_SIZE,
                                 SPARSE_HEAD_DIM,
                                 H,
                                 Waves,
                                 QTiles,
                                 SPARSE_DECODE_PREFETCH,
                                 SPARSE_DECODE_NONTEMPORAL,
                                 KvFmt><<<grid, block, 0, stream>>>(q_idx,
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
                                                                    query_len,
                                                                    rank,
                                                                    world,
                                                                    page_stride,
                                                                    req_xor_mask,
                                                                    fill_shift);
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

template <int Slots, int Waves>
void launch_topk_cp(SPARSE_TOPK_CP_PARAMS)
{
    // One block per (row stride, head). Rows stride by the grid rather than
    // filling it, so the grid stays inside the block array the generation
    // counter is sized for, and inside the residency a spinning merge needs.
    //
    // Held at the full block array even when the batch is shorter, rather than
    // trimmed to it: a block that gets no rows costs a generation and an exit,
    // while a grid that moved with the batch would move which counter times a
    // row along with it, and the tags would stop lining up across calls.
    const int max_rows = kCpMaxBlocks / num_idx_heads;
    dim3 grid(max_rows > 0 ? max_rows : 1, num_idx_heads);
    dim3 block(Waves * kWave);

    pa_sparse_block_topk_cp_kernel<SPARSE_BLOCK_SIZE,
                                   Slots,
                                   SPARSE_TOPK,
                                   SPARSE_TOPK_CP_MERGE_SLOTS,
                                   Waves,
                                   SPARSE_PAGES_PER_BLOCK>
        <<<grid, block, 0, stream>>>(score,
                                     peers,
                                     cp_gen,
                                     topk_idx,
                                     seq_lens,
                                     block_table,
                                     sparse_bt,
                                     sparse_ctx,
                                     num_idx_heads,
                                     total_q,
                                     cand_rows,
                                     score_head_stride,
                                     score_num_stride,
                                     topk_head_stride,
                                     topk_num_stride,
                                     block_table_stride,
                                     sparse_bt_stride,
                                     num_kv_heads,
                                     query_len,
                                     init_blocks,
                                     local_blocks,
                                     rank,
                                     world);
}

} // namespace sparse_attn
} // namespace aiter

// ---------------------------------------------------------------------------
// Per-variant launchers
// ---------------------------------------------------------------------------
#define SPARSE_DECODE_FN(K, H, Q, W) sparse_score_decode_k##K##_h##H##_q##Q##_w##W
#define SPARSE_PREFILL_FN(H, Q) sparse_score_prefill_h##H##_q##Q
#define SPARSE_TOPK_FN(S, W) sparse_topk_s##S##_w##W

#define SPARSE_DECODE_DECLARE(K, H, Q, W) \
    void SPARSE_DECODE_FN(K, H, Q, W)(SPARSE_SCORE_DECODE_PARAMS);
#define SPARSE_PREFILL_DECLARE(H, Q) void SPARSE_PREFILL_FN(H, Q)(SPARSE_SCORE_PREFILL_PARAMS);
#define SPARSE_TOPK_DECLARE(S, W) void SPARSE_TOPK_FN(S, W)(SPARSE_TOPK_PARAMS);

#define SPARSE_DECODE_DEFINE(K, H, Q, W)                           \
    void SPARSE_DECODE_FN(K, H, Q, W)(SPARSE_SCORE_DECODE_PARAMS)  \
    {                                                              \
        launch_score_decode<K, H, Q, W>(SPARSE_SCORE_DECODE_ARGS); \
    }
#define SPARSE_PREFILL_DEFINE(H, Q)                            \
    void SPARSE_PREFILL_FN(H, Q)(SPARSE_SCORE_PREFILL_PARAMS)  \
    {                                                          \
        launch_score_prefill<H, Q>(SPARSE_SCORE_PREFILL_ARGS); \
    }
#define SPARSE_TOPK_DEFINE(S, W) \
    void SPARSE_TOPK_FN(S, W)(SPARSE_TOPK_PARAMS) { launch_topk<S, W>(SPARSE_TOPK_ARGS); }
#define SPARSE_TOPK_CP_FN(S, W) sparse_topk_cp_s##S##_w##W
#define SPARSE_TOPK_CP_DECLARE(S, W) void SPARSE_TOPK_CP_FN(S, W)(SPARSE_TOPK_CP_PARAMS);
#define SPARSE_TOPK_CP_DEFINE(S, W)                     \
    void SPARSE_TOPK_CP_FN(S, W)(SPARSE_TOPK_CP_PARAMS) \
    {                                                   \
        launch_topk_cp<S, W>(SPARSE_TOPK_CP_ARGS);      \
    }

// ---------------------------------------------------------------------------
// scoring & topk kernel instantiation configuration tables
// ---------------------------------------------------------------------------
// (kv format, index heads, query tiles, waves)
// K: kKvFp8 / kKvNvfp4. fp8 keys build one query tile only.
// H is the heads one score pass covers, and the MFMA's 16 columns hold one
// (query token, index head) pair each per query tile. H=4 is what context
// partitioning needs: every rank scores every index head over its own block
// shard, so the pass sees the model's whole head count rather than the
// per-rank one the tensor-parallel split leaves behind.
// clang-format off
#define SPARSE_DECODE_TABLE_KQ(F, K, Q)                                        \
    F(K, 1, Q, 1) F(K, 1, Q, 2) F(K, 1, Q, 4)                                  \
    F(K, 2, Q, 1) F(K, 2, Q, 2) F(K, 2, Q, 4)                                  \
    F(K, 4, Q, 1) F(K, 4, Q, 2) F(K, 4, Q, 4)
// clang-format on
#define SPARSE_DECODE_TABLE_Q1(F) SPARSE_DECODE_TABLE_KQ(F, 0, 1) SPARSE_DECODE_TABLE_KQ(F, 1, 1)
#define SPARSE_DECODE_TABLE_Q2(F) SPARSE_DECODE_TABLE_KQ(F, 1, 2)
#define SPARSE_DECODE_TABLE(F) SPARSE_DECODE_TABLE_Q1(F) SPARSE_DECODE_TABLE_Q2(F)

// (index heads, query tiles); fp8 keys only.
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

// The partitioned chain's shard half walks a row 1/world as long as the
// whole-row selector's, so it needs that same slot range reached from a
// smaller context and is indexed the same way. Its merge half rides along at a
// fixed slot count and adds no axis of its own.
#define SPARSE_TOPK_CP_TABLE(F) SPARSE_TOPK_TABLE(F)

namespace aiter {
namespace sparse_attn {
SPARSE_DECODE_TABLE(SPARSE_DECODE_DECLARE)
SPARSE_PREFILL_TABLE(SPARSE_PREFILL_DECLARE)
// nvfp4 prefill: 1, 2 or 4 index heads, 4 waves x 16 * wave_tiles query rows
// (token, head) per workgroup.
struct nvfp4_prefill_args
{
    int page_stride;   // bytes between pages
    int num_idx_heads; // 1, 2 or 4
    int wave_tiles;    // 2 (heads <= 2), 4 or 8; >= num_idx_heads
    int store_pages;   // pages per score store: 1, or 2 with wave_tiles 8
    // Workgroup order. xcds > 0: 1D grid, each of the xcds XCDs takes a
    // contiguous share of the work. Else the 3D grid, with the group index
    // XORed with key & group_xor and, if req_rot, the request index rotated
    // by key, where key = chunk >> 2. 0 turns each off.
    int group_xor;
    int req_rot;
    int xcds;
    // req_chunks > 0: each request sizes its chunks from its own block count;
    // grid_chunks is then the block-axis grid size.
    int req_chunks;
    int grid_chunks;
};
void sparse_score_prefill_nvfp4(SPARSE_SCORE_PREFILL_PARAMS, const nvfp4_prefill_args& a);
SPARSE_TOPK_TABLE(SPARSE_TOPK_DECLARE)
SPARSE_TOPK_CP_TABLE(SPARSE_TOPK_CP_DECLARE)
} // namespace sparse_attn
} // namespace aiter
