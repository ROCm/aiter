// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "hk/mi35x_v32_fwd_decode_m16x4_fp8_fp8.cuh"
#include "hk/mi35x_v32_fwd_decode_m16x8_fp8_fp8.cuh"
#include "hk/mi3xx_v32_fwd_decode_m16x8_fp8_fp8.cuh"
#include "hk/mla_a16w16_16mx8_32nx1_ps.hpp"
#include "mla.h"

void hk_mla_decode_fwd(torch::Tensor& query,
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
                       torch::Tensor& final_output)
{
    const int32_t num_head = query.size(1);

    const std::string gfx     = get_gpu_arch();
    const int32_t     block_m = num_head * max_seqlen_q;

    // On gfx950, bf16/fp16 (D_qk=576, D_v=512) go through the opus-based a16w16
    // persistent kernel. It uses a fixed M tile of 128 (NUM_WARPS=8); the
    // metadata caps query tokens per work at 128/num_head, so any num_head that
    // divides 128 is supported (full or partial/padded M tiles). This covers
    // e.g. -n 32,4 / 64,2 / 128,1 (full) and -n 64,3 / 128,2 (split into
    // q_len<=128/num_head works).
    const auto q_dtype = query.scalar_type();
    const bool is_a16  = (q_dtype == at::ScalarType::BFloat16 ||
                          q_dtype == at::ScalarType::Half);
    if (gfx == "gfx950" && is_a16 && num_head > 0 && num_head <= 128 &&
        (128 % num_head) == 0)
    {
        hk_mla_a16w16_16mx8_32nx1_ps(query,
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
        return;
    }

    if(block_m == 128)
    {
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
            TORCH_CHECK(false,
                        "hk_mla_decode_fwd: unsupported GPU arch '",
                        gfx,
                        "' (supported: gfx942, gfx950).");
        }
    }
    else if(block_m == 64)
    {
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
            TORCH_CHECK(
                false, "hk_mla_decode_fwd: unsupported GPU arch '", gfx, "' (supported: gfx950).");
        }
    }
    else
    {
        TORCH_CHECK(
            false,
            "hk_mla_decode_fwd requires num_head * max_seqlen_q == 64 or 128, got num_head = ",
            num_head,
            ", max_seqlen_q = ",
            max_seqlen_q);
    }
}
