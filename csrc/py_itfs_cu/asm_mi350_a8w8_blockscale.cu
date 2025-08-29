// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "hip_float8.h"

struct __attribute__((packed)) KernelArgs
{
    void *ptr_O;
    p2 _p0;
    void *ptr_X;
    p2 _p1;
    void *ptr_GU;
    p2 _p2;
    void *ptr_XC;
    p2 _p3;
    void *ptr_XQ;
    p2 _p4;
    void *ptr_GUQ;
    p2 _p5;
    void *ptr_SMQ;
    p2 _p6;
    void *ptr_STP;
    p2 _p7;
    void *ptr_SEP;
    p2 _p8;
    unsigned int dim;
    p3 _p9;
    unsigned int hidden_dim;
    p3 _p10;
    unsigned int token_cnt;
    p3 _p11;
    unsigned int eprt_cnt;
    p3 _p12;
    unsigned int Xs;
    p3 _p13;
    unsigned int GUs;
    p3 _p14;
    unsigned int Os;
    p3 _p15;
    unsigned int eGUs;
    p3 _p16;
    unsigned int eGUQs;
    p3 _p17;
    unsigned int eSMQs;
    p3 _p18;
    unsigned int topk;
    p3 _p19;
    unsigned int splitk;
    p3 _p20;
    unsigned int activation;
    p3 _p21;
    void *ptr_SW;
    p2 _p22;
};

using namespace hip_fp8_impl;
torch::Tensor mi350_a8w8_blockscale_asm(
    torch::Tensor &XQ,      // [M, K]
    torch::Tensor &WQ,      // [N, K] -> [N/128, K*128]
    torch::Tensor &x_scale, // [K/128, M]
    torch::Tensor &w_scale, // [K/128, N/128]
    torch::Tensor &out      // Out:[M, N] fp16
)
{
    int TileM = 128;
    constexpr int TileN = 128;
    constexpr int TileK = 128;

    int m = XQ.size(0);
    int n = out.size(1);
    int k = XQ.size(1);
   if (m <= 32)
       TileM = 32;
    TORCH_CHECK(out.dtype() == torch::ScalarType::BFloat16,
                "mi350 a8w8 blockscale asm only support Half output now!");
    TORCH_CHECK(n % TileN == 0 && k % TileK == 0, 
                "mi350 a8w8 blockscale asm only suuport 128x256x128 tile now!");

    KernelArgs args;
    size_t arg_size = sizeof(args);

    args.ptr_X = (void *)XQ.data_ptr();
    args.ptr_GU = (void *)WQ.data_ptr();
    args.ptr_XQ = (void *)x_scale.data_ptr();
    args.ptr_GUQ = (void *)w_scale.data_ptr();
    args.ptr_O = (void *)out.data_ptr();
    args.dim = k;
    args.hidden_dim = n;
    args.token_cnt = m;
    args.eprt_cnt = 1;
    args.Xs = k * XQ.element_size();
    args.GUs = k * XQ.element_size();
    args.Os = n * 2;
    args.splitk = 0;
    args.activation = 0;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // printf("ptr_X: %p\n", args.ptr_X);
    // printf("ptr_GU: %p\n", args.ptr_GU);
    // printf("ptr_XQ: %p\n", args.ptr_XQ);
    // printf("ptr_GUQ: %p\n", args.ptr_GUQ);
    // printf("args.Xs: %d\n", args.Xs);
    // printf("args.token_cnt: %d\n", args.token_cnt);
    AiterAsmKernel *impl_ptr = nullptr;
    static AiterAsmKernel impl_kenrel_x128("f8_block_scale_mi350_x128", "f8_block_scale_mi350_x128.co");
    static AiterAsmKernel impl_kenrel_x32("f8_block_scale_mi350_x32", "f8_block_scale_mi350_x32.co");
    impl_ptr = (m <= 32)?&impl_kenrel_x32:&impl_kenrel_x128;
    int gdx = (n + TileN*2 - 1) / (TileN*2);
    int gdy = (m + TileM - 1) / TileM;
   // printf("gdx: %d, gdy:%d, \n", gdx, gdy);
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,   // gdx
                             gdy,   // gdy
                             1,     // gdz
                             256,   // bdx: 4 wv64
                             1,     // bdy
                             1,     // bdz
                             stream});                                 

    return out;
}
