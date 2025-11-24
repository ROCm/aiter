// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_hip_common.h"
#include "kittens.cuh"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/python.h>

namespace hk = HipKittens;
namespace ct = ck_tile;

template <typename Q_DTYPE_, typename KV_DTYPE_, int32_t kNumHead_>
struct HkMlaDecodeFwdTraits
{
    using Q_DTYPE  = Q_DTYPE_;
    using KV_DTYPE = KV_DTYPE_;
    using gl_qo    = hk::gl<Q_DTYPE, -1, -1, -1, -1>;
    using gl_kv    = hk::gl<KV_DTYPE, -1, -1, -1, -1>;
    using gl_tmp   = hk::gl<float, -1, -1, -1, -1>

        static constexpr int32_t kNumHead   = kNumHead_;
    static constexpr int32_t kKvLoraRank    = 512;
    static constexpr int32_t kQkRopeHeadDim = 64;
};

template <typename Traits>
struct HkMlaDecodeFwdParams
{
    // inputs
    Traits::gl_qo query;
    Traits::gl_kv kv_buffer;

    // outputs
    Traits::gl_tmp split_data;
    Traits::gl_tmp split_lse;
    Traits::gl_qo final_output;

    // metadata
    const int32_t* p_work_indptr;
    const int32_t* p_work_info_set;
};

template <typename Traits>
void dispatch_mla_decode_fwd_n128(const torch::Tensor& query,
                                  const torch::Tensor& kv_buffer,
                                  const torch::Tensor& qo_indptr,
                                  const torch::Tensor& kv_indptr,
                                  const torch::Tensor& kv_page_indices,
                                  const torch::Tensor& kv_last_page_lens,
                                  const torch::Tensor& work_indptr,
                                  const torch::Tensor& work_info_set,
                                  const int max_seqlen_q,
                                  const float softmax_scale,
                                  torch::Tensor& split_data,
                                  torch::Tensor& split_lse,
                                  torch::Tensor& final_output)
{
    HkMlaDecodeFwdParams<Traits> params;
}

void hk_mi35x_mla_decode_fwd_n128(const torch::Tensor& query,
                                  const torch::Tensor& kv_buffer,
                                  const torch::Tensor& qo_indptr,
                                  const torch::Tensor& kv_indptr,
                                  const torch::Tensor& kv_page_indices,
                                  const torch::Tensor& kv_last_page_lens,
                                  const torch::Tensor& work_indptr,
                                  const torch::Tensor& work_info_set,
                                  const int max_seqlen_q,
                                  const float softmax_scale,
                                  torch::Tensor& split_data,
                                  torch::Tensor& split_lse,
                                  torch::Tensor& final_output)
{
    const bool q_is_fp8 = (query.scalar_type() == at::ScalarType::Float8_e4m3fn) ||
                          (query.scalar_type() == at::ScalarType::Float8_e4m3fnuz);
    const bool kv_is_fp8 = (kv_buffer.scalar_type() == at::ScalarType::Float8_e4m3fn) ||
                           (kv_buffer.scalar_type() == at::ScalarType::Float8_e4m3fnuz);
    const bool q_is_bf16  = (query.scalar_type() == at::ScalarType::BFloat16);
    const bool kv_is_bf16 = (kv_buffer.scalar_type() == at::ScalarType::BFloat16);

    if(q_is_fp8 && kv_is_fp8)
    {
        using Traits = HkMlaDecodeFwdTraits<kittens::fp8e4m3, kittens::fp8e4m3, 128>;
        dispatch_mla_decode_fwd_n128<Traits>(query,
                                             kv_buffer,
                                             qo_indptr,
                                             kv_indptr,
                                             kv_page_indices,
                                             kv_last_page_lens,
                                             work_indptr,
                                             work_info_set,
                                             max_seqlen_q,
                                             softmax_scale,
                                             split_data,
                                             split_lse,
                                             final_output);
    }
    else if(q_is_bf16 && kv_is_bf16)
    {
        using Traits = HkMlaDecodeFwdTraits<kittens::bf16, kittens::bf16, 128>;
        dispatch_mla_decode_fwd_n128<Traits>(query,
                                             kv_buffer,
                                             qo_indptr,
                                             kv_indptr,
                                             kv_page_indices,
                                             kv_last_page_lens,
                                             work_indptr,
                                             work_info_set,
                                             max_seqlen_q,
                                             softmax_scale,
                                             split_data,
                                             split_lse,
                                             final_output);
    }
    else
    {
        TORCH_CHECK(false,
                    "hk_mi35x_mla_decode_fwd_n128 doesn't support q type ",
                    toString(query.scalar_type()),
                    " and kv type",
                    toString(kv_buffer.scalar_type()),
                    ".");
    }
}
