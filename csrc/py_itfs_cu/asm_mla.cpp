// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"

struct __attribute__((packed)) KernelArgs
{
    void *ptr_R;
    p2 _p0;
    void *ptr_LSE;
    p2 _p1;
    void *ptr_Q;
    p2 _p2;
    void *ptr_KV;
    p2 _p3;
    void *ptr_LTP;
    p2 _p4;
    void *ptr_LTD;
    p2 _p5;
    void *ptr_LTL;
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
    void *ptr_QTP;
    p2 _p18;
};

void mla_stage1_asm_fwd(torch::Tensor &Q,                 //   [num_seqs, num_heads, head_size]
                        torch::Tensor &KV,                //   [num_page, page_size, num_kv_heads, head_size]
                        torch::Tensor &kv_indptr,         //   [batch_size+1]
                        torch::Tensor &kv_page_indices,   //   [num_page_used]
                        torch::Tensor &kv_last_page_lens, //   [batch_size]
                        float softmax_scale,
                        // following are output
                        torch::Tensor &splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
                        torch::Tensor &splitLse   //[batch_size, num_kv_splits, num_heads,  1]

)
{
    int num_seqs = Q.size(0);
    int num_heads = Q.size(1);
    int head_size = Q.size(2);
    int page_size = KV.size(1);
    int num_kv_heads = KV.size(2);
    int kv_split = splitData.size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    int stride_Q = Q.stride(0) * Q.itemsize();
    int stride_Page = KV.stride(0) * KV.itemsize();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_R = splitData.data_ptr();
    args.ptr_LSE = splitLse.data_ptr();
    args.ptr_Q = Q.data_ptr();
    args.ptr_KV = KV.data_ptr();
    args.ptr_LTP = kv_indptr.data_ptr();
    args.ptr_LTD = kv_page_indices.data_ptr();
    args.ptr_LTL = kv_last_page_lens.data_ptr();
    args.scalar = softmax_scale;
    args.s_MQA = gqa_ratio;
    args.s_kv_split = kv_split;
    args.s_Q_Bs = stride_Q;
    args.s_Bs = stride_Page;
    args.s_log2_plen = log2_page;
    // std::cout << "scalar: " << args.scalar << " s_MQA:" << args.s_MQA << " s_kv_split:" << args.s_kv_split << " s_Q_Bs:" << args.s_Q_Bs << " s_Bs:" << args.s_Bs << " s_log2_plen:" << args.s_log2_plen << std::endl;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel *impl_ptr = nullptr;
    TORCH_CHECK(Q.is_contiguous(),
                __func__, ":only support Q.is_contiguous() for now");
    TORCH_CHECK(num_kv_heads == 1,
                __func__, ":only support num_kv_heads==1 for now");
    TORCH_CHECK(head_size == KV.size(3),
                __func__, ":only support head_size == KV.size(3) for now");
    if (Q.dtype() == at::ScalarType::BFloat16)
    {
        static AiterAsmKernel impl_a16w16_bf16("mla_stage1_a16w16_bf16", "/mla/mla_stage1_a16w16_bf16.co");
        impl_ptr = &impl_a16w16_bf16;
    }

    TORCH_CHECK(impl_ptr != nullptr,
                __func__, ": unsupport current input type");
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             (gqa_ratio + 15) / 16, // gdx
                             num_seqs,              // gdy
                             kv_split,              // gdz
                             256,                   // bdx: 4 wv64
                             1,                     // bdy
                             1,                     // bdz
                             stream});
}

void mla_prefill_asm_fwd(torch::Tensor &Q,                 //   [num_seqs, num_heads, head_size]
                         torch::Tensor &KV,                //   [num_page, page_size, num_kv_heads, head_size]
                         torch::Tensor &qo_indptr,         //   [batch_size+1]
                         torch::Tensor &kv_indptr,         //   [batch_size+1]
                         torch::Tensor &kv_page_indices,   //   [num_page_used]
                         torch::Tensor &kv_last_page_lens, //   [batch_size]
                         int max_seqlen_q,
                         float softmax_scale,
                         // following are output
                         torch::Tensor &splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
                         torch::Tensor &splitLse   //[batch_size, num_kv_splits, num_heads,  1]

)
{
    int sub_Q = 128;
    int batch = kv_indptr.size(0) - 1;
    int num_heads = Q.size(1);
    int head_size = Q.size(2);
    int page_size = KV.size(1);
    int num_kv_heads = KV.size(2);
    int kv_split = splitData.size(1);
    const int gqa_ratio = num_heads / num_kv_heads;

    int stride_Q = Q.stride(0) * Q.itemsize();
    int stride_Page = KV.stride(0) * KV.itemsize();
    uint32_t log2_page = (uint32_t)log2f(page_size);

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_R = splitData.data_ptr();
    args.ptr_LSE = splitLse.data_ptr();
    args.ptr_Q = Q.data_ptr();
    args.ptr_KV = KV.data_ptr();
    args.ptr_LTP = kv_indptr.data_ptr();
    args.ptr_LTD = kv_page_indices.data_ptr();
    args.ptr_LTL = kv_last_page_lens.data_ptr();
    args.ptr_QTP = qo_indptr.data_ptr();
    args.scalar = softmax_scale;
    args.s_MQA = gqa_ratio;
    args.s_kv_split = kv_split;
    args.s_Q_Bs = stride_Q;
    args.s_Bs = stride_Page;
    args.s_log2_plen = log2_page;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel *impl_ptr = nullptr;
    TORCH_CHECK(Q.is_contiguous(),
                __func__, ":only support Q.is_contiguous() for now");
    TORCH_CHECK(gqa_ratio == 16,
                __func__, ":only support num_q_heads/num_kv_heads==16 for now");
    TORCH_CHECK(num_kv_heads == 1,
                __func__, ":only support num_kv_heads==1 for now");
    TORCH_CHECK(head_size == KV.size(3),
                __func__, ":only support head_size == KV.size(3) for now");
    if (Q.dtype() == at::ScalarType::BFloat16)
    {
        static AiterAsmKernel impl_a16w16_bf16("mla_pfl_a16w16_bf16_causal", "/mla/mla_pfl_a16w16_bf16_causal.co");
        impl_ptr = &impl_a16w16_bf16;
    }

    TORCH_CHECK(impl_ptr != nullptr,
                __func__, ": unsupport current input type");
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