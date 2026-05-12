// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "mla.h"
#include "mla_hk.h"

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
                           aiter_tensor_t& final_output)
{
    (void)query;
    (void)query_rope;
    (void)kv_buffer;
    (void)kv_buffer_rope;
    (void)qo_indptr;
    (void)kv_indptr;
    (void)kv_page_indices;
    (void)kv_last_page_lens;
    (void)work_indptr;
    (void)work_info_set;
    (void)max_seqlen_q;
    (void)softmax_scale;
    (void)split_output;
    (void)split_lse;
    (void)final_output;
}
