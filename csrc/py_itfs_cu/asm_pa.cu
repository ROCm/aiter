// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_O;
    p2 _p0;
    void* ptr_Q;
    p2 _p1;
    void* ptr_K;
    p2 _p2;
    void* ptr_V;
    p2 _p3;
    void* ptr_BT;
    p2 _p4;
    void* ptr_CL;
    p2 _p5;
    void* ptr_KQ;
    p2 _p6;
    void* ptr_VQ;
    p2 _p7;
    float sclg2e;
    p3 _p12;
    unsigned int mblk;
    p3 _p13;
    unsigned int kv_nheads;
    p3 _p14;
    unsigned int Qs;
    p3 _p15;
    unsigned int Bs;
    p3 _p16;
    unsigned int KVs;
    p3 _p17;
    unsigned int GQA;
    p3 _p18;
    void* ptr_QTP;
    p2 _p19;
};

const float f_log2E = log2f(expf(1));

torch::Tensor pa_fwd(torch::Tensor& Q, //   [num_seqs, num_heads, head_size]
                     torch::Tensor& K, //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                     torch::Tensor& V, //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
                     torch::Tensor& block_tables, //   [num_seqs, max_num_blocks_per_seq]
                     torch::Tensor& context_lens, //   [num_seqs]
                     int max_num_blocks,
                     int max_qlen                           = 1,
                     std::optional<torch::Tensor> K_QScale  = std::nullopt,
                     std::optional<torch::Tensor> V_QScale  = std::nullopt,
                     std::optional<torch::Tensor> out_      = std::nullopt,
                     std::optional<torch::Tensor> qo_indptr = std::nullopt,
                     std::optional<int> high_precision      = 1)
{
    torch::Tensor output = out_.value_or(torch::empty_like(Q));
    int batch            = context_lens.size(0);
    // int max_num_blocks = block_tables.size(1);
    int num_heads       = Q.size(1);
    int head_size       = Q.size(2);
    int num_kv_heads    = K.size(1);
    int block_size      = K.size(3);
    const int gqa_ratio = num_heads / num_kv_heads;
    TORCH_CHECK(block_size == 16, __func__, " for now only support block_size == 16");

    int dim            = head_size;
    int stride_Q       = Q.stride(0) * Q.itemsize();
    int stride_KV_head = block_size * dim * K.itemsize();
    int stride_KV_blk  = stride_KV_head * num_kv_heads;
    float k_log2e      = f_log2E;
    float k_scalar     = sqrt(dim);
    k_scalar           = (float)((double)k_log2e / (double)k_scalar);

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_O      = output.data_ptr();
    args.ptr_Q      = Q.data_ptr();
    args.ptr_K      = K.data_ptr();
    args.ptr_V      = V.data_ptr();
    args.ptr_BT     = block_tables.data_ptr();
    args.ptr_CL     = context_lens.data_ptr();
    if(K_QScale)
    {
        args.ptr_KQ = K_QScale.value().data_ptr();
        args.ptr_VQ = V_QScale.value().data_ptr();
    }
    else
    {
        args.ptr_KQ = nullptr;
        args.ptr_VQ = nullptr;
    }
    args.sclg2e    = k_scalar;
    args.mblk      = max_num_blocks;
    args.kv_nheads = num_kv_heads;
    args.Qs        = stride_Q;
    args.Bs        = stride_KV_blk;
    args.KVs       = stride_KV_head;
    args.GQA       = gqa_ratio;
    args.ptr_QTP   = qo_indptr ? qo_indptr.value().data_ptr() : nullptr;
    // std::cout << "sclg2e: " << args.sclg2e << " mblk:" << args.mblk << "
    // kv_nheads:" << args.kv_nheads << " Qs:" << args.Qs << " Bs:" << args.Bs <<
    // " KVs:" << args.KVs << std::endl;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel* impl_ptr = nullptr;
    if(qo_indptr && max_qlen > 1)
    {
        if(K.dtype() == at::ScalarType::BFloat16)
        {
            if(gqa_ratio <= 8)
            {
                static AiterAsmKernel impl_bf16_noquant_a16w16_gqa8_qlen_msk1(
                    "_ZN5aiter32pa_bf16_noquant_a16w16_gqa8_qlen_msk1E",
                    "/pa/pa_bf16_noquant_a16w16_gqa8_qlen_msk1.co");
                impl_ptr = &impl_bf16_noquant_a16w16_gqa8_qlen_msk1;
            }
            // else if (gqa_ratio <= 16)
            // {
            //     static AiterAsmKernel
            //     impl_a16w16_1tg_g8_f8_gqa16_qlen_msk1("pa_bf16_noquant_a16w16_gqa8_qlen_msk1",
            //     "/pa/pa_bf16_noquant_a16w16_gqa8_qlen_msk1.co"); impl_ptr =
            //     &impl_a16w16_1tg_g8_f8_gqa16_qlen_msk1;
            // }
            else
            {
                TORCH_CHECK(false,
                            __func__,
                            ": gqa_ratio only support less 16 on bf16 asm pa with "
                            "qo_indptr !!!");
            }
        }
        else
        {
            TORCH_CHECK(Q.dtype() == at::ScalarType::BFloat16 && K_QScale && K.dtype() == torch_fp8,
                        __func__,
                        ": qo_indptr only support bf16 asm pa with fp8 kv cache");

            if(gqa_ratio <= 8)
            {
                static AiterAsmKernel impl_a16w8_1tg_g8_f8_gqa8_qlen_msk1(
                    "_ZN5aiter35pa_bf16_pertokenFp8_a16w8_gqa8_qlen_msk1E",
                    "/pa/pa_bf16_pertokenFp8_a16w8_gqa8_qlen_msk1.co");
                impl_ptr = &impl_a16w8_1tg_g8_f8_gqa8_qlen_msk1;
            }
            else if(gqa_ratio <= 16)
            {
                static AiterAsmKernel impl_a16w16_1tg_g8_f8_gqa16_qlen_msk1(
                    "_ZN5aiter36pa_bf16_pertokenFp8_a16w8_gqa16_qlen_msk1E",
                    "/pa/pa_bf16_pertokenFp8_a16w8_gqa16_qlen_msk1.co");
                impl_ptr = &impl_a16w16_1tg_g8_f8_gqa16_qlen_msk1;
            }
            else
            {
                TORCH_CHECK(false,
                            __func__,
                            ": gqa_ratio only support less 16 on bf16 asm pa with "
                            "qo_indptr !!!");
            }
        }
    }
    else if(K_QScale)
    {
        if(Q.dtype() == at::ScalarType::Half)
        {
            if(K.dtype() == at::ScalarType::Byte || K.dtype() == at::ScalarType::Char)
            {
                static AiterAsmKernel impl_a16w8_f16_i8("pa_a16w8_2tg_g8_i8",
                                                        "pa_a16w8_f16_2tg_g8_i8.co");
                impl_ptr = &impl_a16w8_f16_i8;
            }
            else if(K.dtype() == torch_fp8)
            {
                if(high_precision.value() == 0)
                {
                    static AiterAsmKernel impl_a16w8_f16_f8(
                        "_ZN5aiter32pa_fp16_pertokenFp8_a16w8_2tg_g8E",
                        "/pa/pa_fp16_pertokenFp8_a16w8_2tg_g8.co");
                    impl_ptr = &impl_a16w8_f16_f8;
                }
                else if(high_precision.value() == 1)
                {
                    static AiterAsmKernel impl_a16w8_2tg_g8_f8_q_fp16_tail_bf16(
                        "_ZN5aiter45pa_bf16_pertokenFp8_a16w8_g8_q_fp16_tail_bf16E",
                        "/pa/pa_bf16_pertokenFp8_a16w8_g8_q_fp16_tail_bf16.co");
                    impl_ptr = &impl_a16w8_2tg_g8_f8_q_fp16_tail_bf16;
                }
                else
                {
                    TORCH_CHECK(false,
                                __func__,
                                ": high_precision value only support (0, 1) grades on "
                                "fp16 asm pa for fp8 kv cache !!!");
                }
            }
        }
        else if(Q.dtype() == at::ScalarType::BFloat16)
        {
            if(K.dtype() == at::ScalarType::Byte || K.dtype() == at::ScalarType::Char)
            {
                static AiterAsmKernel impl_a16w8_b16_i8(
                    "_ZN5aiter33pa_bf16_pertokenInt8_a16w8_2tg_g8E",
                    "/pa/pa_bf16_pertokenInt8_a16w8_2tg_g8.co");
                impl_ptr = &impl_a16w8_b16_i8;
            }
            else if(K.dtype() == torch_fp8)
            {
                if(high_precision.value() == 0)
                {
                    static AiterAsmKernel impl_a16w8_b16_f8(
                        "_ZN5aiter32pa_bf16_pertokenFp8_a16w8_2tg_g8E",
                        "/pa/pa_bf16_pertokenFp8_a16w8_2tg_g8.co");
                    impl_ptr = &impl_a16w8_b16_f8;
                }
                else if(high_precision.value() == 1)
                {
                    static AiterAsmKernel impl_a16w8_b16_f8_tail_bf16(
                        "_ZN5aiter42pa_bf16_pertokenFp8_a16w8_2tg_g8_tail_bf16E",
                        "/pa/pa_bf16_pertokenFp8_a16w8_2tg_g8_tail_bf16.co");
                    impl_ptr = &impl_a16w8_b16_f8_tail_bf16;
                }
                else if(high_precision.value() == 2)
                {
                    static AiterAsmKernel impl_a16w8_b16_f8_gemm1_bf16(
                        "_ZN5aiter43pa_bf16_pertokenFp8_a16w8_2tg_g8_gemm1_bf16E",
                        "/pa/pa_bf16_pertokenFp8_a16w8_2tg_g8_gemm1_bf16.co");
                    impl_ptr = &impl_a16w8_b16_f8_gemm1_bf16;
                }
                else
                {
                    TORCH_CHECK(false,
                                __func__,
                                ": high_precision value only support (0, 1, 2) grades on "
                                "bf16 asm pa for fp8 kv cache !!!");
                }
            }
        }
    }
    else
    {
        TORCH_CHECK(Q.is_contiguous() || Q.dtype() == at::ScalarType::BFloat16,
                    __func__,
                    ":a16w16_f16 only support Q.is_contiguous() for now");
        TORCH_CHECK(num_kv_heads == 1 || Q.dtype() == at::ScalarType::BFloat16,
                    __func__,
                    ":a16w16_f16 only support num_kv_heads==1, for now");
        if(Q.dtype() == at::ScalarType::Half)
        {
            static AiterAsmKernel impl_a16w16_f16("pa_kernel_func", "pa_a16w16_f16.co");
            impl_ptr = &impl_a16w16_f16;
        }
        else if(Q.dtype() == at::ScalarType::BFloat16)
        {
            static AiterAsmKernel impl_a16w16_b16("_ZN5aiter22pa_bf16_noquant_a16w16E",
                                                  "/pa/pa_bf16_noquant_a16w16.co");
            impl_ptr = &impl_a16w16_b16;
        }
    }
    TORCH_CHECK(impl_ptr != nullptr, __func__, ": unsupport current Q_type:", Q.scalar_type());

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             num_kv_heads, // gdx
                             batch,        // gdy
                             1,            // gdz
                             256,          // bdx: 4 wv64
                             1,            // bdy
                             1,            // bdz
                             stream});
    return output;
}
