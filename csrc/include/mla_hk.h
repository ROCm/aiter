// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Torch-free entry points for the HipKittens MLA decode kernels.
//   v3.2: csrc/kernels/mla/hk_v32_decode_fwd.cu
//   v4.0: csrc/kernels/mla/hk_v40_decode_fwd.cu

#include "aiter_tensor.h"
#include <optional>

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
                           const aiter_tensor_t& kv_page_indices,
                           const aiter_tensor_t& kv_last_page_lens,
                           const aiter_tensor_t& work_indptr,
                           const aiter_tensor_t& work_info_set,
                           const int max_seqlen_q,
                           const float softmax_scale,
                           aiter_tensor_t& split_output,
                           aiter_tensor_t& split_lse,
                           aiter_tensor_t& final_output,
                           std::optional<aiter_tensor_t> attn_sink);
