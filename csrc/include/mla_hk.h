// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Torch-free entry points for the HipKittens MLA decode kernels.
//   v3.2: csrc/kernels/mla/hk_v32_decode_fwd.cu
//   v4.0: csrc/kernels/mla/hk_v40_decode_fwd.cu

#include "aiter_tensor.h"

void hk_mla_v32_decode_fwd(aiter_tensor_t& query,
                           aiter_tensor_t& kv_buffer,
                           const aiter_tensor_t& qo_indptr,
                           const aiter_tensor_t& kv_indptr,
                           const aiter_tensor_t& kv_page_indices,
                           const aiter_tensor_t& kv_last_page_lens,
                           const aiter_tensor_t& work_indptr,
                           const aiter_tensor_t& work_info_set,
                           const int max_seqlen_q,
                           const float softmax_scale,
                           aiter_tensor_t& split_output,
                           aiter_tensor_t& split_lse,
                           aiter_tensor_t& final_output);

// V4.0 MLA decode entry. Q/KV are split:
//   query        : [total_q, nhead, V4_DIM_QK_PACKED=576]  FP8  (NOPE 448
//                  + duplicated E8M0 scale 16 + zero pad 112 per token)
//   query_rope   : [total_q, nhead, V4_DIM_ROPE=64]        BF16
//   kv_buffer    : [num_page, page_size, 1, 576]           FP8  (same packing)
//   kv_buffer_rope: [num_page, page_size, 1, 64]           BF16
// Constraints: (max_seqlen_q * nhead) == 128, FP8 NOPE, BF16 ROPE.
void hk_mla_v40_decode_fwd(aiter_tensor_t& query,
                           aiter_tensor_t& query_rope,
                           aiter_tensor_t& kv_buffer,
                           aiter_tensor_t& kv_buffer_rope,
                           const aiter_tensor_t& qo_indptr,
                           const aiter_tensor_t& kv_indptr,
                           const aiter_tensor_t& kv_page_indices,
                           const aiter_tensor_t& kv_last_page_lens,
                           const aiter_tensor_t& work_indptr,
                           const aiter_tensor_t& work_info_set,
                           const int max_seqlen_q,
                           const float softmax_scale,
                           aiter_tensor_t& split_output,
                           aiter_tensor_t& split_lse,
                           aiter_tensor_t& final_output);

// QManager8to16bitsV1 standalone unit test entry. Runs only the Q loader
// (no QK / softmax / PV / epilogue) and dumps the result to two tensors.
//   query        : [1, 128, 576]  FP8 packed Q (1 token, 128 heads)
//   query_rope   : [1, 128, 64]   BF16
//   q_vgpr_out   : [8, 16, 256]   BF16 -- VGPR half (Q[:, 0:256])
//                  laid out as [warp_idx, head_in_warp (0..15), feat (0..255)]
//   q_lds_out    : [8, 16, 256]   BF16 -- LDS  half (Q[:, 256:512])
//                  same indexing as q_vgpr_out
void hk_mla_v40_qmanager_v1_unit_test(aiter_tensor_t& query,
                                      aiter_tensor_t& query_rope,
                                      aiter_tensor_t& q_vgpr_out,
                                      aiter_tensor_t& q_lds_out);

// Diagnostic for QManager8to16bitsV1 Phase-1 ladder: replays the chunk 0/1/2/3
// producer ladder with checkpoint LDS dumps. dump_out layout:
//   [warp_idx (0..7), slot (0..3), 512 bf16] = 4 KB/warp
// slot 0 = buf 0 after chunk-0 producer
// slot 1 = buf 1 after chunk-1 producer
// slot 2 = buf 0 after chunk-0 consumer + chunk-2 producer
// slot 3 = buf 1 after chunk-1 consumer + chunk-3 producer
void hk_mla_v40_qmanager_v1_p1_ladder_probe(aiter_tensor_t& query,
                                            aiter_tensor_t& query_rope,
                                            aiter_tensor_t& dump_out);
