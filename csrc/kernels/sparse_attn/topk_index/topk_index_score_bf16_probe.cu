// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// bf16 index-K FEASIBILITY PROBE -- not a product. Unregistered, bound by no
// Python entry, and the operator still refuses a bf16 cache.
// OPUS_IDX_SCORE_ALLOW_BF16_K is defined here and nowhere else.
#define OPUS_IDX_SCORE_ALLOW_BF16_K 1

#include "topk_index_score.hpp"

#include "aiter_ctypes_error.h"
#include "aiter_hip_common.h"

AITER_CTYPES_ERROR_DECL;

namespace aiter {
namespace sparse_attn {
namespace {

// The bf16-K __global__. Same template, same MFMA, same fp32 accumulator and
// the same fp32 NaN-seeded fold -- only the A-fragment dtype changes, so the
// NaN-seed derivation (kernels.cuh's "if this FOLD is ever retyped" note) is
// untouched and still holds. k_to_bf16 becomes the identity.
template <int H, int Q, int AUX_K>
__global__ __launch_bounds__(kOpusNumWarps* kOpusWarpSize, 2) void
opus_decode_index_score_bf16_kernel(const opus_decode_score_args a)
{
    opus_index_score::decode_index_score_impl<
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
        opus::bf16_t,
#else
        uint8_t, // stub pass, as in the shipped wrapper
#endif
        OPUS_IDX_HEAD_DIM,
        OPUS_IDX_BLOCK_SIZE,
        H,
        Q,
        AUX_K>(a);
}

template <int H, int Q, int AUX_K>
void launch_bf16_probe(OPUS_IDX_SCORE_PARAMS)
{
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
    opus_decode_index_score_bf16_kernel<H, Q, AUX_K><<<grid, block, 0, stream>>>(a);
}

} // namespace
} // namespace sparse_attn
} // namespace aiter

#define OPUS_BF16_PROBE_DISPATCH(H, Q, A)                                       \
    if(num_idx_heads == (H) && query_len == (Q) && aux_k == (A))                \
    {                                                                           \
        aiter::sparse_attn::launch_bf16_probe<H, Q, A>(OPUS_IDX_SCORE_ARGS);    \
        return;                                                                 \
    }

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(topk_index_score_bf16_probe,
                                    (size_t q_idx_ptr,
                                     size_t key_cache_idx_ptr,
                                     size_t score_ptr,
                                     size_t block_table_ptr,
                                     size_t seq_lens_ptr,
                                     long long q_numel,
                                     long long key_cache_numel,
                                     long long score_numel,
                                     long long block_table_numel,
                                     int batch,
                                     int num_chunks,
                                     int chunk_blocks,
                                     long long stride_q_n,
                                     long long stride_q_h,
                                     long long stride_ik_blk,
                                     long long stride_s_h,
                                     long long stride_s_b,
                                     long long stride_bt_b,
                                     float sm_scale,
                                     int num_idx_heads,
                                     int query_len,
                                     int aux_k,
                                     hipStream_t stream),
                                    (q_idx_ptr,
                                     key_cache_idx_ptr,
                                     score_ptr,
                                     block_table_ptr,
                                     seq_lens_ptr,
                                     q_numel,
                                     key_cache_numel,
                                     score_numel,
                                     block_table_numel,
                                     batch,
                                     num_chunks,
                                     chunk_blocks,
                                     stride_q_n,
                                     stride_q_h,
                                     stride_ik_blk,
                                     stride_s_h,
                                     stride_s_b,
                                     stride_bt_b,
                                     sm_scale,
                                     num_idx_heads,
                                     query_len,
                                     aux_k,
                                     stream))
{
    // Types MUST match OPUS_IDX_SCORE_PARAMS exactly: score is float*, and the
    // block table / seq_lens are const int*, not void*.
    const void* q_idx         = reinterpret_cast<const void*>(q_idx_ptr);
    const void* key_cache_idx = reinterpret_cast<const void*>(key_cache_idx_ptr);
    float* score              = reinterpret_cast<float*>(score_ptr);
    const int* block_table    = reinterpret_cast<const int*>(block_table_ptr);
    const int* seq_lens       = reinterpret_cast<const int*>(seq_lens_ptr);

    AITER_CHECK(batch > 0 && num_chunks > 0 && chunk_blocks > 0,
                "opus bf16 probe: bad launch geometry");
    AITER_CHECK(aux_k == 0 || aux_k == 3, "opus bf16 probe: aux_k must be 0 or 3");

    OPUS_BF16_PROBE_DISPATCH(1, 1, 0)
    OPUS_BF16_PROBE_DISPATCH(1, 1, 3)
    OPUS_BF16_PROBE_DISPATCH(1, 4, 0)
    OPUS_BF16_PROBE_DISPATCH(1, 4, 3)
    OPUS_BF16_PROBE_DISPATCH(4, 1, 0)
    OPUS_BF16_PROBE_DISPATCH(4, 1, 3)
    OPUS_BF16_PROBE_DISPATCH(4, 4, 0)
    OPUS_BF16_PROBE_DISPATCH(4, 4, 3)
    // The probe deliberately does NOT follow the widened table: it exists to
    // answer one feasibility question on the cells that already had evidence,
    // and it is unregistered anyway.
    AITER_CHECK(false, "opus bf16 probe: no build for this (H, Q, aux_k)");
}
