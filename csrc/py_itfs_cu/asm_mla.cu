// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_R;
    p2 _p0;
    void* ptr_LSE;
    p2 _p1;
    void* ptr_Q;
    p2 _p2;
    void* ptr_KV;
    p2 _p3;
    void* ptr_LTP;
    p2 _p4;
    void* ptr_LTD;
    p2 _p5;
    void* ptr_LTL;
    p2 _p6;
    float scalar;
    p3 _p12;
    unsigned int s_MQA;
    p3 _p13;
    unsigned int s_kv_split;
    p3 _p14;
    unsigned int s_Q_Bs;
    p3 _p15;
    unsigned int s_Bs;
    p3 _p16;
    unsigned int s_log2_plen;
    p3 _p17;
    void* ptr_QTP;
    p2 _p18;
    void* ptr_STP;
    p2 _p19;
    void* ptr_RP;
    p2 _p20;
    void* ptr_QSCALE;
    p2 _p21;
    void* ptr_KVSCALE;
    p2 _p22;
};

void mla_decode_stage1_asm_fwd(
    torch::Tensor& Q,                    //   [num_seqs, num_heads, head_size]
    torch::Tensor& KV,                   //   [num_page, page_size, num_kv_heads, head_size]
    torch::Tensor& qo_indptr,            //   [batch_size+1]
    torch::Tensor& kv_indptr,            //   [batch_size+1]
    torch::Tensor& kv_page_indices,      //   [num_page_used]
    torch::Tensor& kv_last_page_lens,    //   [batch_size]
    std::optional<torch::Tensor>& num_kv_splits_indptr,   //   metadata
    std::optional<torch::Tensor>& work_meta_data,         //   metadata addr
    std::optional<torch::Tensor>& work_indptr,            //   metadata
    std::optional<torch::Tensor>& work_info_set,          //   [batch_size+1]
    int max_seqlen_q,
    float softmax_scale,
    // following are output
    torch::Tensor& splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
    torch::Tensor& splitLse,  //[batch_size, num_kv_splits, num_heads,  1]
    torch::Tensor& output,    //[batch_size, num_heads, v_head_dim]
    std::optional<torch::Tensor> q_scale  = std::nullopt, //   [1]
    std::optional<torch::Tensor> kv_scale = std::nullopt  //   [1]
)
{
    int batch           = qo_indptr.size(0) - 1;
    int num_heads       = Q.size(1);
    int head_size       = Q.size(2);
    int page_size       = KV.size(1);
    int num_kv_heads    = KV.size(2);
    int kv_split        = splitData.size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    bool persistent = !num_kv_splits_indptr.has_value();

    int stride_Q       = Q.stride(0) * Q.itemsize() * max_seqlen_q;
    int stride_Page    = KV.stride(0) * KV.itemsize();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args;
    size_t arg_size  = sizeof(args);
    args.ptr_R       = splitData.data_ptr();
    args.ptr_LSE     = splitLse.data_ptr();
    args.ptr_Q       = Q.data_ptr();
    args.ptr_KV      = KV.data_ptr();
    args.ptr_LTP     = kv_indptr.data_ptr();
    args.ptr_LTD     = kv_page_indices.data_ptr();
    args.ptr_LTL     = kv_last_page_lens.data_ptr();
    args.ptr_QTP     = qo_indptr.data_ptr();
    args.scalar      = softmax_scale;
    args.s_MQA       = gqa_ratio * max_seqlen_q;
    args.s_kv_split  = kv_split;
    args.s_Q_Bs      = stride_Q;
    args.s_Bs        = stride_Page;
    args.s_log2_plen = log2_page;

    if (persistent)
    {
        if (work_meta_data.has_value())
        {
            args.ptr_STP = work_meta_data.value().data_ptr();
        }
        else
        {
            assert(work_indptr.has_value() && work_info_set.has_value());
            assert(work_indptr.value().data_ptr() != nullptr && work_info_set.value().data_ptr() != nullptr);

            uint64_t* persistent_meta_data = new uint64_t[10];
            persistent_meta_data[0] = (uint64_t)work_indptr.value().data_ptr();
            persistent_meta_data[1] = (uint64_t)work_info_set.value().data_ptr();
            uint32_t* dev_PS_META_DATA;

            unsigned long buf_size_META = 10 * sizeof(uint64_t);
            hipMalloc(&dev_PS_META_DATA, buf_size_META);
            hipMemcpy(dev_PS_META_DATA, persistent_meta_data, buf_size_META, hipMemcpyHostToDevice);

            args.ptr_STP = dev_PS_META_DATA;
        }
    }
    else
    {
        args.ptr_STP = num_kv_splits_indptr.value().data_ptr();
    }
	args.ptr_RP = output.data_ptr();

    // std::cout << "mla args" << std::endl;
    // std::cout << "ptr_R: " << args.ptr_R << std::endl;
    // std::cout << "ptr_LSE: " << args.ptr_LSE << std::endl;
    // std::cout << "ptr_Q: " << args.ptr_Q << std::endl;
    // std::cout << "ptr_KV: " << args.ptr_KV << std::endl;
    // std::cout << "ptr_LTP: " << args.ptr_LTP << std::endl;
    // std::cout << "ptr_LTD: " << args.ptr_LTD << std::endl;
    // std::cout << "ptr_LTL: " << args.ptr_LTL << std::endl;
    // std::cout << "scalar: " << args.scalar << std::endl;
    // std::cout << "s_MQA: " << args.s_MQA << std::endl;
    // std::cout << "s_kv_split: " << args.s_kv_split << std::endl;
    // std::cout << "s_Q_Bs: " << args.s_Q_Bs << std::endl;
    // std::cout << "s_Bs: " << args.s_Bs << std::endl;
    // std::cout << "s_log2_plen: " << args.s_log2_plen << std::endl;
    // std::cout << "ptr_RP: " << args.ptr_RP << std::endl;
    // std::cout << "ptr_QTP: " << args.ptr_QTP << std::endl;
    // std::cout << "ptr_STP: " << args.ptr_STP << std::endl;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel* impl_ptr = nullptr;
    TORCH_CHECK(Q.is_contiguous(), __func__, ":only support Q.is_contiguous() for now");
    TORCH_CHECK(num_kv_heads == 1, __func__, ":only support num_kv_heads==1 for now");
    TORCH_CHECK(head_size == KV.size(3), __func__, ":only support head_size == KV.size(3) for now");
    int sub_Q;
    if(Q.dtype() == at::ScalarType::BFloat16)
    {
        if(gqa_ratio == 128)
        {
            sub_Q = 128;
            static AiterAsmKernel impl_a16w16_bf16_subQ128(
                "_ZN5aiter41mla_dec_stage1_bf16_a16w16_subQ128_mqa128E",
                "/mla/mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co");
            impl_ptr = &impl_a16w16_bf16_subQ128;
        }
        else if(gqa_ratio == 16)
        {
            if(persistent)
            {
                if(max_seqlen_q <= 4)
                {
                    sub_Q = 128;
                    static AiterAsmKernel impl_a16w16_bf16_ps(
                        "_ZN5aiter42mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_psE",
                        "/mla/mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps.co");
                    impl_ptr = &impl_a16w16_bf16_ps;
                }
            }
            else if(max_seqlen_q == 1)
            {
                sub_Q = 16;
                static AiterAsmKernel impl_a16w16_bf16(
                    "_ZN5aiter39mla_dec_stage1_bf16_a16w16_subQ16_mqa16E",
                    "/mla/mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co");
                impl_ptr = &impl_a16w16_bf16;
            }
            else if(max_seqlen_q <= 4)
            {
                sub_Q = 128;
                static AiterAsmKernel impl_a16w16_bf16(
                    "_ZN5aiter39mla_a16w16_qh16_m16x4_n16x1_coex0_mask1E",
                    "/mla/mla_a16w16_qh16_m16x4_n16x1_coex0_mask1.co");
                impl_ptr = &impl_a16w16_bf16;
            }
            else
            {
                sub_Q = 128;
                static AiterAsmKernel impl_a16w16_bf16(
                    "_ZN5aiter39mla_a16w16_qh16_m32x4_n16x1_coex0_mask1E",
                    "/mla/mla_a16w16_qh16_m32x4_n16x1_coex0_mask1.co");
                impl_ptr = &impl_a16w16_bf16;
            }
        }
    }
    else if(Q.dtype() == at::ScalarType::Float8_e4m3fnuz) // at::ScalarType::Float8_e4m3fnuz in mi300
    {
        assert(q_scale.has_value() && kv_scale.has_value());
        assert(q_scale.value().data_ptr() != nullptr && kv_scale.value().data_ptr() != nullptr);
        args.ptr_QSCALE  = q_scale.value().data_ptr();
        args.ptr_KVSCALE = kv_scale.value().data_ptr();

        if(gqa_ratio == 16)
        {
            if(persistent)
            {
                if(max_seqlen_q == 1)
                {
                    sub_Q = 128;
                    static AiterAsmKernel impl_fp8(
                        "_ZN5aiter36mla_a8w8_qh16_qseqlen1_gqaratio16_psE",
                        "/mla/mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co");
                    impl_ptr = &impl_fp8;
                }
                else if(max_seqlen_q == 2)
                {
                    sub_Q = 128;
                    static AiterAsmKernel impl_fp8(
                        "_ZN5aiter36mla_a8w8_qh16_qseqlen2_gqaratio16_psE",
                        "/mla/mla_a8w8_qh16_qseqlen2_gqaratio16_ps.co");
                    impl_ptr = &impl_fp8;
                }
                else if(max_seqlen_q <= 4)
                {
                    // assert(false);
                    sub_Q = 128;
                    static AiterAsmKernel impl_fp8(
                        "_ZN5aiter36mla_a8w8_qh16_qseqlen4_gqaratio16_psE",
                        "/mla/mla_a8w8_qh16_qseqlen4_gqaratio16_ps.co");
                    impl_ptr = &impl_fp8;
                }
                else
                {
                    TORCH_CHECK(false, __func__, ":only support fp8 mla decoding for qo_len <= 4");
                }
            }
            else
            {
                TORCH_CHECK(false, __func__, ":only support persistent fp8 mla");
            }
        }
        else if(gqa_ratio == 128)
        {
            if(persistent)
            {
                // assert(false);
                sub_Q = 128;
                static AiterAsmKernel impl_fp8(
                    "_ZN5aiter28mla_a8w8_qh16_gqaratio128_psE",
                    "/mla/mla_a8w8_qh16_gqaratio128_ps.co");
                impl_ptr = &impl_fp8;
            }
            else
            {
                sub_Q = 128;
                static AiterAsmKernel impl_fp8(
                    "_ZN5aiter25mla_a8w8_qh16_gqaratio128E",
                    "/mla/mla_a8w8_qh16_gqaratio128.co");
                impl_ptr = &impl_fp8;
            }
        }

    }

    TORCH_CHECK(impl_ptr != nullptr, __func__, ": unsupport current Q_type:", Q.scalar_type());

    int bdx = 256;
    int gdx = (max_seqlen_q * gqa_ratio + sub_Q - 1) / sub_Q;
    int gdy = batch;
    int gdz = kv_split;

    if(persistent)
    {
        gdx = work_indptr.value().size(0) - 1;
        gdy = 1;
        gdz = 1;
    }
    // printf("gdx: %d \n", gdx);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,       // gdx
                             gdy,       // gdy
                             gdz,       // gdz
                             256,       // bdx: 4 wv64
                             1,         // bdy
                             1,         // bdz
                             stream});
}

void mla_prefill_asm_fwd(
    torch::Tensor& Q,                 //   [num_seqs, num_heads, head_size]
    torch::Tensor& KV,                //   [num_page, page_size, num_kv_heads, head_size]
    torch::Tensor& qo_indptr,         //   [batch_size+1]
    torch::Tensor& kv_indptr,         //   [batch_size+1]
    torch::Tensor& kv_page_indices,   //   [num_page_used]
    torch::Tensor& kv_last_page_lens, //   [batch_size]
    int max_seqlen_q,
    float softmax_scale,
    // following are output
    torch::Tensor& splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
    torch::Tensor& splitLse   //[batch_size, num_kv_splits, num_heads,  1]

)
{
    int sub_Q           = 128;
    int batch           = kv_indptr.size(0) - 1;
    int num_heads       = Q.size(1);
    int head_size       = Q.size(2);
    int page_size       = KV.size(1);
    int num_kv_heads    = KV.size(2);
    int kv_split        = splitData.size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    int stride_Q       = Q.stride(0) * Q.itemsize();
    int stride_Page    = KV.stride(0) * KV.itemsize();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args;
    size_t arg_size  = sizeof(args);
    args.ptr_R       = splitData.data_ptr();
    args.ptr_LSE     = splitLse.data_ptr();
    args.ptr_Q       = Q.data_ptr();
    args.ptr_KV      = KV.data_ptr();
    args.ptr_LTP     = kv_indptr.data_ptr();
    args.ptr_LTD     = kv_page_indices.data_ptr();
    args.ptr_LTL     = kv_last_page_lens.data_ptr();
    args.ptr_QTP     = qo_indptr.data_ptr();
    args.scalar      = softmax_scale;
    args.s_MQA       = gqa_ratio;
    args.s_kv_split  = kv_split;
    args.s_Q_Bs      = stride_Q;
    args.s_Bs        = stride_Page;
    args.s_log2_plen = log2_page;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel* impl_ptr = nullptr;
    TORCH_CHECK(Q.is_contiguous(), __func__, ":only support Q.is_contiguous() for now");
    TORCH_CHECK(gqa_ratio == 16 || gqa_ratio == 128,
                __func__,
                ":only support num_q_heads/num_kv_heads==16 or 128 for now");
    TORCH_CHECK(num_kv_heads == 1, __func__, ":only support num_kv_heads==1 for now");
    TORCH_CHECK(head_size == KV.size(3), __func__, ":only support head_size == KV.size(3) for now");
    if(Q.dtype() == at::ScalarType::BFloat16)
    {
        if(gqa_ratio == 16)
        {
            static AiterAsmKernel impl_a16w16_bf16(
                "_ZN5aiter39mla_pfl_bf16_a16w16_causal_subQ16_mqa16E",
                "/mla/mla_pfl_bf16_a16w16_causal_subQ16_mqa16.co");
            impl_ptr = &impl_a16w16_bf16;
        }
        else if(gqa_ratio == 128)
        {
            static AiterAsmKernel impl_a16w16_bf16(
                "_ZN5aiter41mla_pfl_bf16_a16w16_causal_subQ128_mqa128E",
                "/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co");
            impl_ptr = &impl_a16w16_bf16;
        }
    }

    TORCH_CHECK(impl_ptr != nullptr, __func__, ": unsupport current Q_type:", Q.scalar_type());
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             (max_seqlen_q * gqa_ratio + sub_Q - 1) / sub_Q, // gdx
                             batch,                                          // gdy
                             kv_split,                                       // gdz
                             256,                                            // bdx: 4 wv64
                             1,                                              // bdy
                             1,                                              // bdz
                             stream});
}
