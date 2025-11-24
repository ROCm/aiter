// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_hip_common.h"
#include "kittens.cuh"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/python.h>

namespace hk  = kittens;
namespace ckt = ck_tile;

template <typename q_t_, typename kv_t_, int32_t kQoNumHead_>
struct HkMlaDecodeFwdTraits
{
    static constexpr int32_t kQoNumHead     = kQoNumHead_;
    static constexpr int32_t kKvNumHead     = 1;
    static constexpr int32_t kKvLoraRank    = 512;
    static constexpr int32_t kQkNopeHeadDim = kKvLoraRank;
    static constexpr int32_t kQkRopeHeadDim = 64;
    static constexpr int32_t kQkHeadDim     = kQkNopeHeadDim + kQkRopeHeadDim;
    static constexpr int32_t kVoHeadDim     = kKvLoraRank;
    static constexpr int32_t kPageSize      = 1;
    static constexpr int32_t kNumWarps      = 8;
    static constexpr int32_t kNumThreads    = kNumWarps * ckt::get_warp_size();
    static constexpr int32_t kOccupancy     = 1;

    using q_t  = q_t_;
    using kv_t = kv_t_;
    using gl_q = hk::gl<q_t, 1, -1, kQoNumHead, kQkHeadDim>; // [1, #batch*#seqlen, #head, 576]
    using gl_kv =
        hk::gl<kv_t, -1, kPageSize, kKvNumHead, kQkHeadDim>;    // [#page, page_size, #head_kv, 576]
    using gl_o    = hk::gl<q_t, 1, -1, kQoNumHead, kVoHeadDim>; // [1, #batch*#seqlen, #head, 512]
    using gl_so   = hk::gl<float, 1, -1, kQoNumHead, kVoHeadDim>; // [1, #partial_slots, #head, 512]
    using gl_slse = hk::gl<float, 1, -1, kQoNumHead, 1>;          // [1, #partial_slots, #head, 1]
};

template <typename Traits>
struct HkMlaDecodeFwdParams
{
    // inputs
    Traits::gl_q query;
    Traits::gl_kv kv_buffer;

    // outputs
    Traits::gl_o final_output;
    Traits::gl_so split_output;
    Traits::gl_slse split_lse;

    // metadata
    const int32_t* p_work_indptr;
    const int32_t* p_work_info_set;
};

template <typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads, Traits::kOccupancy) __attribute__((
    amdgpu_num_vgpr(64))) void kn_ml_decode_fwd_n128(HkMlaDecodeFwdParams<Traits> params)
{
    extern __shared__ int32_t p_lds[];
    hk::shared_allocator al(p_lds);

    if(blockIdx.x < 2)
    {
        if(threadIdx.x < 4)
        {
            printf("[hkmla-dbg][%d %d]\n", blockIdx.x, threadIdx.x);
        }
    }
}

template <typename Traits>
void dispatch_mla_decode_fwd_n128(torch::Tensor& query,
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
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const hipStream_t stream = at::hip::getCurrentHIPStream();

    HkMlaDecodeFwdParams<Traits> params = {
        hk::make_gl<typename Traits::gl_q>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query.data_ptr())),
            1,
            query.size(0),
            Traits::kQoNumHead,
            Traits::kQkHeadDim),
        hk::make_gl<typename Traits::gl_kv>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kv_buffer.data_ptr())),
            kv_buffer.size(0),
            Traits::kPageSize,
            Traits::kKvNumHead,
            Traits::kQkHeadDim),
        hk::make_gl<typename Traits::gl_o>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(final_output.data_ptr())),
            1,
            final_output.size(0),
            Traits::kQoNumHead,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_so>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_output.data_ptr())),
            1,
            split_output.size(0),
            Traits::kQoNumHead,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_slse>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_lse.data_ptr())),
            1,
            split_lse.size(0),
            Traits::kQoNumHead,
            1),
        // metadata
        work_indptr.data_ptr<int32_t>(),
        work_info_set.data_ptr<int32_t>()};

    const dim3 grid        = dim3(dev_prop.multiProcessorCount);
    const int32_t lds_size = dev_prop.maxSharedMemoryPerMultiProcessor / Traits::kOccupancy;

    kn_ml_decode_fwd_n128<Traits><<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
}

void hk_mi35x_mla_decode_fwd_n128(torch::Tensor& query,
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
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(final_output));

    const bool q_is_fp8 = (query.scalar_type() == at::ScalarType::Float8_e4m3fn) ||
                          (query.scalar_type() == at::ScalarType::Float8_e4m3fnuz);
    const bool kv_is_fp8 = (kv_buffer.scalar_type() == at::ScalarType::Float8_e4m3fn) ||
                           (kv_buffer.scalar_type() == at::ScalarType::Float8_e4m3fnuz);
    const bool q_is_bf16  = (query.scalar_type() == at::ScalarType::BFloat16);
    const bool kv_is_bf16 = (kv_buffer.scalar_type() == at::ScalarType::BFloat16);

    if(q_is_fp8 && kv_is_fp8)
    {
        using Traits = HkMlaDecodeFwdTraits<hk::fp8e4m3, hk::fp8e4m3, 128>;
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
                                             split_output,
                                             split_lse,
                                             final_output);
    }
    else if(q_is_bf16 && kv_is_bf16)
    {
        using Traits = HkMlaDecodeFwdTraits<hk::bf16, hk::bf16, 128>;
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
                                             split_output,
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
