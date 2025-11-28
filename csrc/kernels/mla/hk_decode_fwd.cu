// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "mla.h"
#include "hk/mi35x_decode_fwd_n128.cuh"

void hk_mla_decode_fwd(
    torch::Tensor& query,
    torch::Tensor& kv_buffer,
    const torch::Tensor& qo_indptr,
    const torch::Tensor& kv_indptr,
    const torch::Tensor& kv_page_indices,
    const torch::Tensor& kv_last_page_lens,
    const torch::Tensor& work_indptr,
    const torch::Tensor& work_info_set,
    const int max_seqlen_q,
    const float softmax_scale,
    torch::Tensor& split_output,
    torch::Tensor& split_lse,
    torch::Tensor& final_output,
    std::optional<torch::Tensor>& dbg_tr)
{
    const int32_t num_head = query.size(1);

    if (num_head == 128)
    {
        hk_mi35x_mla_decode_fwd_n128(
            query,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_page_indices,
            kv_last_page_lens,
            work_indptr,
            work_info_set,
            max_seqlen_q,
            softmax_scale,
            split_output,
            split_lse,
            final_output,
            dbg_tr);
    }
}
