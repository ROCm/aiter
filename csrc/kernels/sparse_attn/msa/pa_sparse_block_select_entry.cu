// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// aiter_ctypes_error.h needs aiter_detail from the headers this pulls in, so
// keep it first (blank line stops clang-format from sorting the two together).
#include "pa_sparse_block_select.hpp"

#include "aiter_ctypes_error.h"

// Per-.so TLS error storage + aiter_get_last_error / aiter_clear_last_error
// exports, so the unsupported-shape checks below surface as a Python
// RuntimeError instead of aborting the worker process.
AITER_CTYPES_ERROR_DEF

// The shapes that are fixed for the whole build rather than dispatched on.
#define SPARSE_CHECK_SHAPE(pass)                                                    \
    AITER_CHECK(head_dim == SPARSE_HEAD_DIM,                                        \
                pass ": head_dim ",                                                 \
                head_dim,                                                           \
                " unsupported; one MFMA covers the whole head dim, so it must be ", \
                SPARSE_HEAD_DIM);                                                   \
    AITER_CHECK(block_size == SPARSE_BLOCK_SIZE,                                    \
                pass ": block_size ",                                               \
                block_size,                                                         \
                " is not built; this build carries block_size ",                    \
                SPARSE_BLOCK_SIZE)

// The round-robin block shard, global = local * world + rank. Checked here
// rather than left to the kernel because a rank outside its world silently
// scores the wrong blocks instead of failing.
#define SPARSE_CHECK_SHARD(pass)                                     \
    AITER_CHECK(world >= 1 && rank >= 0 && rank < world,             \
                pass ": rank ",                                      \
                rank,                                                \
                " is not a shard of a world of ",                    \
                world)

inline bool sparse_arch_supported()
{
    static const bool v = (get_gpu_arch() == "gfx950");
    return v;
}

#define SPARSE_CHECK_ARCH(pass)                         \
    AITER_CHECK(sparse_arch_supported(),                \
                pass ": requires gfx950 architecture; " \
                     "the running device is ",          \
                get_gpu_arch())

#define SPARSE_DECODE_DISPATCH(H, W)                                          \
    if(num_idx_heads == (H) && num_waves == (W))                              \
    {                                                                         \
        aiter::sparse_attn::SPARSE_DECODE_FN(H, W)(SPARSE_SCORE_DECODE_ARGS); \
        return;                                                               \
    }

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(pa_sparse_block_score_decode,
                                    (size_t q_idx_ptr,
                                     size_t key_cache_idx_ptr,
                                     size_t score_ptr,
                                     size_t block_table_ptr,
                                     size_t seq_lens_ptr,
                                     int num_reqs,
                                     int num_q_tiles,
                                     int block_table_stride,
                                     int score_head_stride,
                                     int score_num_stride,
                                     int num_chunks,
                                     int init_blocks,
                                     int local_blocks,
                                     int query_len,
                                     int rank,
                                     int world,
                                     int block_size,
                                     int head_dim,
                                     int num_idx_heads,
                                     int num_waves,
                                     hipStream_t stream),
                                    (q_idx_ptr,
                                     key_cache_idx_ptr,
                                     score_ptr,
                                     block_table_ptr,
                                     seq_lens_ptr,
                                     num_reqs,
                                     num_q_tiles,
                                     block_table_stride,
                                     score_head_stride,
                                     score_num_stride,
                                     num_chunks,
                                     init_blocks,
                                     local_blocks,
                                     query_len,
                                     rank,
                                     world,
                                     block_size,
                                     head_dim,
                                     num_idx_heads,
                                     num_waves,
                                     stream))
{
    SPARSE_CHECK_ARCH("pa_sparse_block_score_decode");
    SPARSE_CHECK_SHAPE("pa_sparse_block_score_decode");
    SPARSE_CHECK_SHARD("pa_sparse_block_score_decode");

    const auto* q_idx         = reinterpret_cast<const uint8_t*>(q_idx_ptr);
    const auto* key_cache_idx = reinterpret_cast<const uint8_t*>(key_cache_idx_ptr);
    auto* score               = reinterpret_cast<float*>(score_ptr);
    const auto* block_table   = reinterpret_cast<const int*>(block_table_ptr);
    const auto* seq_lens      = reinterpret_cast<const int*>(seq_lens_ptr);

    SPARSE_DECODE_TABLE(SPARSE_DECODE_DISPATCH)

    AITER_CHECK(false,
                "pa_sparse_block_score_decode: no build for num_idx_heads=",
                num_idx_heads,
                " num_waves=",
                num_waves);
}

#define SPARSE_PREFILL_DISPATCH(H, Q)                                           \
    if(num_idx_heads == (H) && q_tiles == (Q))                                  \
    {                                                                           \
        aiter::sparse_attn::SPARSE_PREFILL_FN(H, Q)(SPARSE_SCORE_PREFILL_ARGS); \
        return;                                                                 \
    }

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(pa_sparse_block_score_prefill,
                                    (size_t q_idx_ptr,
                                     size_t key_cache_idx_ptr,
                                     size_t score_ptr,
                                     size_t block_table_ptr,
                                     size_t cu_seqlens_q_ptr,
                                     size_t seq_lens_ptr,
                                     int num_reqs,
                                     int num_q_groups,
                                     int block_table_stride,
                                     int score_head_stride,
                                     int score_num_stride,
                                     int num_chunks,
                                     int chunk_blocks,
                                     int init_blocks,
                                     int local_blocks,
                                     int block_size,
                                     int head_dim,
                                     int num_idx_heads,
                                     int num_waves,
                                     int q_tiles,
                                     hipStream_t stream),
                                    (q_idx_ptr,
                                     key_cache_idx_ptr,
                                     score_ptr,
                                     block_table_ptr,
                                     cu_seqlens_q_ptr,
                                     seq_lens_ptr,
                                     num_reqs,
                                     num_q_groups,
                                     block_table_stride,
                                     score_head_stride,
                                     score_num_stride,
                                     num_chunks,
                                     chunk_blocks,
                                     init_blocks,
                                     local_blocks,
                                     block_size,
                                     head_dim,
                                     num_idx_heads,
                                     num_waves,
                                     q_tiles,
                                     stream))
{
    SPARSE_CHECK_ARCH("pa_sparse_block_score_prefill");
    SPARSE_CHECK_SHAPE("pa_sparse_block_score_prefill");
    // The workgroup stages one page for all of its waves, so the wave count is
    // part of that mapping and not a variant axis.
    AITER_CHECK(num_waves == SPARSE_PREFILL_WAVES,
                "pa_sparse_block_score_prefill: num_waves ",
                num_waves,
                " is not built; this build carries num_waves ",
                SPARSE_PREFILL_WAVES);

    const auto* q_idx         = reinterpret_cast<const uint8_t*>(q_idx_ptr);
    const auto* key_cache_idx = reinterpret_cast<const uint8_t*>(key_cache_idx_ptr);
    auto* score               = reinterpret_cast<float*>(score_ptr);
    const auto* block_table   = reinterpret_cast<const int*>(block_table_ptr);
    const auto* cu_seqlens_q  = reinterpret_cast<const int*>(cu_seqlens_q_ptr);
    const auto* seq_lens      = reinterpret_cast<const int*>(seq_lens_ptr);

    SPARSE_PREFILL_TABLE(SPARSE_PREFILL_DISPATCH)

    AITER_CHECK(false,
                "pa_sparse_block_score_prefill: no build for num_idx_heads=",
                num_idx_heads,
                " q_tiles=",
                q_tiles);
}

// ---------------------------------------------------------------------------
// Top-k
// ---------------------------------------------------------------------------
#define SPARSE_TOPK_DISPATCH(S, W)                                  \
    if(slots == (S) && num_waves == (W))                            \
    {                                                               \
        aiter::sparse_attn::SPARSE_TOPK_FN(S, W)(SPARSE_TOPK_ARGS); \
        return;                                                     \
    }

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(pa_sparse_block_topk,
                                    (size_t score_ptr,
                                     size_t topk_idx_ptr,
                                     size_t seq_lens_ptr,
                                     size_t num_valid_pages_ptr,
                                     size_t block_table_ptr,
                                     size_t row_req_id_ptr,
                                     size_t kv_lens_ptr,
                                     size_t sparse_bt_ptr,
                                     size_t sparse_ctx_ptr,
                                     int num_idx_heads,
                                     int total_q,
                                     int score_head_stride,
                                     int score_num_stride,
                                     int topk_head_stride,
                                     int topk_num_stride,
                                     int block_table_stride,
                                     int sparse_bt_stride,
                                     int num_kv_heads,
                                     int query_len,
                                     int block_size,
                                     int topk,
                                     int slots,
                                     int num_waves,
                                     int pages_per_block,
                                     hipStream_t stream),
                                    (score_ptr,
                                     topk_idx_ptr,
                                     seq_lens_ptr,
                                     num_valid_pages_ptr,
                                     block_table_ptr,
                                     row_req_id_ptr,
                                     kv_lens_ptr,
                                     sparse_bt_ptr,
                                     sparse_ctx_ptr,
                                     num_idx_heads,
                                     total_q,
                                     score_head_stride,
                                     score_num_stride,
                                     topk_head_stride,
                                     topk_num_stride,
                                     block_table_stride,
                                     sparse_bt_stride,
                                     num_kv_heads,
                                     query_len,
                                     block_size,
                                     topk,
                                     slots,
                                     num_waves,
                                     pages_per_block,
                                     stream))
{
    AITER_CHECK(block_size == SPARSE_BLOCK_SIZE,
                "pa_sparse_block_topk: block_size ",
                block_size,
                " is not built; this build carries block_size ",
                SPARSE_BLOCK_SIZE);
    AITER_CHECK(topk == SPARSE_TOPK,
                "pa_sparse_block_topk: topk ",
                topk,
                " is not built; this build carries topk ",
                SPARSE_TOPK);
    AITER_CHECK(pages_per_block == SPARSE_PAGES_PER_BLOCK,
                "pa_sparse_block_topk: pages_per_block ",
                pages_per_block,
                " is not built; this build carries pages_per_block ",
                SPARSE_PAGES_PER_BLOCK);

    const auto* score           = reinterpret_cast<const float*>(score_ptr);
    auto* topk_idx              = reinterpret_cast<int*>(topk_idx_ptr);
    const auto* seq_lens        = reinterpret_cast<const int*>(seq_lens_ptr);
    const auto* num_valid_pages = reinterpret_cast<const int*>(num_valid_pages_ptr);
    const auto* block_table     = reinterpret_cast<const int*>(block_table_ptr);
    const auto* row_req_id      = reinterpret_cast<const int*>(row_req_id_ptr);
    const auto* kv_lens         = reinterpret_cast<const int*>(kv_lens_ptr);
    auto* sparse_bt             = reinterpret_cast<int*>(sparse_bt_ptr);
    auto* sparse_ctx            = reinterpret_cast<int*>(sparse_ctx_ptr);

    SPARSE_TOPK_TABLE(SPARSE_TOPK_DISPATCH)

    AITER_CHECK(
        false, "pa_sparse_block_topk: no build for slots=", slots, " num_waves=", num_waves);
}

// ---------------------------------------------------------------------------
// Top-k, context-parallel
// ---------------------------------------------------------------------------
// Built here rather than in a file of its own, unlike the three passes above.
// Those each carry a table of their own and are worth an object each; this one
// shares the whole-row selector's slot table, and putting it beside the
// dispatch that is its only caller costs nothing -- the instantiations are the
// build time either way, and this unit was already parsing the templates.
namespace aiter {
namespace sparse_attn {
SPARSE_TOPK_CP_TABLE(SPARSE_TOPK_CP_DEFINE)
} // namespace sparse_attn
} // namespace aiter

#define SPARSE_TOPK_CP_DISPATCH(S, W)                                     \
    if(slots == (S) && num_waves == (W))                                  \
    {                                                                     \
        aiter::sparse_attn::SPARSE_TOPK_CP_FN(S, W)(SPARSE_TOPK_CP_ARGS); \
        return;                                                           \
    }

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(pa_sparse_block_topk_cp,
                                    (size_t score_ptr,
                                     size_t peer_cand_ptrs,
                                     size_t cp_gen_ptr,
                                     size_t topk_idx_ptr,
                                     size_t seq_lens_ptr,
                                     size_t block_table_ptr,
                                     size_t sparse_bt_ptr,
                                     size_t sparse_ctx_ptr,
                                     int num_idx_heads,
                                     int total_q,
                                     int cand_rows,
                                     int score_head_stride,
                                     int score_num_stride,
                                     int topk_head_stride,
                                     int topk_num_stride,
                                     int block_table_stride,
                                     int sparse_bt_stride,
                                     int num_kv_heads,
                                     int query_len,
                                     int init_blocks,
                                     int local_blocks,
                                     int rank,
                                     int world,
                                     int block_size,
                                     int topk,
                                     int slots,
                                     int num_waves,
                                     hipStream_t stream),
                                    (score_ptr,
                                     peer_cand_ptrs,
                                     cp_gen_ptr,
                                     topk_idx_ptr,
                                     seq_lens_ptr,
                                     block_table_ptr,
                                     sparse_bt_ptr,
                                     sparse_ctx_ptr,
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
                                     world,
                                     block_size,
                                     topk,
                                     slots,
                                     num_waves,
                                     stream))
{
    AITER_CHECK(block_size == SPARSE_BLOCK_SIZE,
                "pa_sparse_block_topk_cp: block_size ",
                block_size,
                " is not built; this build carries block_size ",
                SPARSE_BLOCK_SIZE);
    AITER_CHECK(topk == SPARSE_TOPK,
                "pa_sparse_block_topk_cp: topk ",
                topk,
                " is not built; this build carries topk ",
                SPARSE_TOPK);
    SPARSE_CHECK_SHARD("pa_sparse_block_topk_cp");
    AITER_CHECK(world <= aiter::sparse_attn::kCpMaxRanks,
                "pa_sparse_block_topk_cp: world ",
                world,
                " exceeds the ",
                aiter::sparse_attn::kCpMaxRanks,
                " ranks the peer mapping holds");
    // The merge half rides the shard pass's launch at a fixed slot count, so
    // the candidates every shard sends have to fit one wave's lanes.
    AITER_CHECK(world * topk <= SPARSE_TOPK_CP_MERGE_SLOTS * aiter::sparse_attn::kWave,
                "pa_sparse_block_topk_cp: world ",
                world,
                " x topk ",
                topk,
                " exceeds the ",
                SPARSE_TOPK_CP_MERGE_SLOTS * aiter::sparse_attn::kWave,
                " candidates the fused merge holds");
    AITER_CHECK(num_idx_heads % world == 0,
                "pa_sparse_block_topk_cp: ",
                num_idx_heads,
                " index heads do not divide over a world of ",
                world);
    // The buffer is strided by its allocation so that an address keeps one
    // meaning across calls, which only works if the batch fits inside it.
    AITER_CHECK(total_q <= cand_rows,
                "pa_sparse_block_topk_cp: ",
                total_q,
                " rows exceed the ",
                cand_rows,
                " the candidate buffer was allocated for");

    const auto* score       = reinterpret_cast<const float*>(score_ptr);
    auto* cp_gen            = reinterpret_cast<uint32_t*>(cp_gen_ptr);
    auto* topk_idx          = reinterpret_cast<int*>(topk_idx_ptr);
    const auto* seq_lens    = reinterpret_cast<const int*>(seq_lens_ptr);
    const auto* block_table = reinterpret_cast<const int*>(block_table_ptr);
    auto* sparse_bt         = reinterpret_cast<int*>(sparse_bt_ptr);
    auto* sparse_ctx        = reinterpret_cast<int*>(sparse_ctx_ptr);

    // A host array, so the peer table is built here and reaches the kernel by
    // value rather than as a pointer it would have to chase on the device.
    const auto* cand_tbl = reinterpret_cast<const uint64_t*>(peer_cand_ptrs);
    aiter::sparse_attn::CpPeers peers{};
    for(int i = 0; i < world; ++i)
        peers.cand[i] = reinterpret_cast<uint64_t*>(cand_tbl[i]);

    SPARSE_TOPK_CP_TABLE(SPARSE_TOPK_CP_DISPATCH)

    AITER_CHECK(false,
                "pa_sparse_block_topk_cp: no build for slots=",
                slots,
                " num_waves=",
                num_waves);
}
