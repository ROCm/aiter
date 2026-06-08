// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "hk/mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1.cuh"
#include "mla.h"
#include "mla_hk.h"

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
                           std::optional<aiter_tensor_t> attn_sink)
{
    const int32_t num_head = query.size(1);
    const std::string gfx  = get_gpu_arch();

    if((num_head * max_seqlen_q == 128) && (gfx == "gfx950"))
    {
        hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(query,
                                                               query_rope,
                                                               kv_buffer,
                                                               kv_buffer_rope,
                                                               qo_indptr,
                                                               kv_page_indices,
                                                               kv_last_page_lens,
                                                               work_indptr,
                                                               work_info_set,
                                                               max_seqlen_q,
                                                               softmax_scale,
                                                               split_output,
                                                               split_lse,
                                                               final_output,
                                                               attn_sink);
    }
    else
    {
        AITER_CHECK(false,
                    "hk_mla_v40_decode_fwd: only gfx950 with num_head * max_seqlen_q == 128 "
                    "is supported; got gfx='",
                    gfx,
                    "', num_head=",
                    num_head,
                    ", max_seqlen_q=",
                    max_seqlen_q);
    }
}
