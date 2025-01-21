// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "aiter_hip_common.h"

// start to prepare the input and output buffer
struct __attribute__((packed)) KernelArgs
{
    void *ptr_c;
    p2 _p0;
    void *ptr_a;
    p2 _p1;
    void *ptr_b;
    p2 _p2;
    void *ptr_sa;
    p2 _p3;
    void *ptr_sb;
    p2 _p4;
    void *ptr_bias;
    p2 _p5;
    // float alpha;
    unsigned int m;
    p3 _p12;
    unsigned int n;
    p3 _p13;
    unsigned int k;
    p3 _p14;
    unsigned int lda;
    p3 _p15;
    unsigned int ldb;
    p3 _p16;
    unsigned int ldc;
    p3 _p17;
    unsigned int ks;
    p3 _p18;
};

torch::Tensor gemm_a8w8_asm(torch::Tensor &A,       // A:[M, K] i8
                            torch::Tensor &B,       // B:[N, K] i8 -> shuffle layout(32,16)
                            torch::Tensor &A_scale, // A_scale:[M, 1] f32
                            torch::Tensor &B_scale, // B_scale:[1, N] f32
                            torch::Tensor &out,     // Out:[M, N] bf16
                            torch::Tensor &bias,    // bias:[1, N] f32
                            std::optional<int> sub_m = 128,
                            std::optional<int> sub_n = 128,
                            std::optional<int> pad_a = 0,
                            std::optional<int> pad_b = 0,
                            std::optional<int> pad_c = 0,
                            std::optional<int> splitK = 0)
{
    TORCH_CHECK(out.dtype() == torch::ScalarType::BFloat16,
                "GEMM A8W8 asm only support BFloat16 output now!");
    int m = A.size(0);
    int n = out.size(1);
    int k = A.size(1);
    int stride_a = k + pad_a.value();
    int stride_b = k + pad_b.value();
    int stride_c = n + pad_c.value();
    stride_c = stride_c * sizeof(uint16_t);
    int ks = splitK.value();

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_c = (void *)out.data_ptr();
    args.ptr_a = (void *)A.data_ptr();
    args.ptr_b = (void *)B.data_ptr();
    args.ptr_sa = (void *)A_scale.data_ptr();
    args.ptr_sb = (void *)B_scale.data_ptr();
    args.ptr_bias = (void *)bias.data_ptr();

    args.m = m;
    args.n = n;
    args.k = k;
    args.lda = stride_a;
    args.ldb = stride_b;
    args.ldc = stride_c;
    args.ks = ks;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    static AiterAsmKernel splitK_impl("gemm_kernel_func", "gemm_a8w8_m128_splitK.co");
    static AiterAsmKernel noSplitK_impl("gemm_kernel_func", "gemm_a8w8_m128_noSplitK.co");
    AiterAsmKernel *impl_ptr = &noSplitK_impl;
    if (ks > 0)
        impl_ptr = &splitK_impl;

    int sub_m_v = sub_m.value();
    int sub_n_v = sub_n.value();
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             (n / sub_n_v) << ks,           // gdx
                             ((m + sub_m_v - 1) / sub_m_v), // gdy
                             1,                             // gdz
                             256,                           // bdx: 4 wv64
                             1,                             // bdy
                             1,                             // bdz
                             stream});
    return out;
}
