// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "hk/mi35x_v32_fwd_decode_m16x4_fp8_fp8.cuh"
#include "hk/mi35x_v32_fwd_decode_m16x8_fp8_fp8.cuh"
#include "hk/mi3xx_v32_fwd_decode_m16x8_fp8_fp8.cuh"
#include "mla.h"

void hk_mla_decode_fwd(aiter_tensor_t& query,
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
                       aiter_tensor_t& final_output)
{
    const int32_t num_head = query.size(1);

    if((num_head * max_seqlen_q) == 128)
    {
        const std::string gfx = get_gpu_arch();
        if(gfx == "gfx942")
        {
            hk_mi3xx_mla_v32_fwd_decode_m16x8_fp8_fp8(query,
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
                                                      final_output);
        }
        else if(gfx == "gfx950")
        {
            hk_mi35x_mla_v32_fwd_decode_m16x8_fp8_fp8(query,
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
                                                      final_output);
        }
        else
        {
            AITER_CHECK(false,
                        "hk_mla_decode_fwd: unsupported GPU arch '",
                        gfx,
                        "' (supported: gfx942, gfx950).");
        }
    }
    else if((num_head * max_seqlen_q) == 64)
    {
        const std::string gfx = get_gpu_arch();
        if(gfx == "gfx950")
        {
            hk_mi35x_mla_v32_fwd_decode_m16x4_fp8_fp8(query,
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
                                                      final_output);
        }
        else
        {
            AITER_CHECK(
                false, "hk_mla_decode_fwd: unsupported GPU arch '", gfx, "' (supported: gfx950).");
        }
    }
    else
    {
        AITER_CHECK(
            false,
            "hk_mla_decode_fwd requires num_head * max_seqlen_q == 64 or 128, got num_head = ",
            num_head,
            ", max_seqlen_q = ",
            max_seqlen_q);
    }
}
