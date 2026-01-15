// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "aiter_hip_common.h"
#include "asm_fp8gemm_blockscale_mi350_code_objects.hpp"
#include "hip_float8.h"

struct __attribute__((packed)) KernelArgs
{
    void *ptr_C;
    p2 _p0;
    void *ptr_X;
    p2 _p1;
    void *ptr_W;
    p2 _p2;
    void *ptr_XC;
    p2 _p3;
    void *ptr_XQ;
    p2 _p4;
    void *ptr_WQ;
    p2 _p5;
    void *ptr_tmp0;
    p2 _p6;
    void *ptr_tmp1;
    p2 _p7;
    void *ptr_tmp2;
    p2 _p8;
    unsigned int K;
    p3 _p9;
    unsigned int N;
    p3 _p10;
    unsigned int M;
    p3 _p11;
    unsigned int eprt_cnt;
    p3 _p12;
    unsigned int Xs;
    p3 _p13;
    unsigned int Ws;
    p3 _p14;
    unsigned int Cs;
    p3 _p15;
    unsigned int tmp0;
    p3 _p16;
    unsigned int tmp1;
    p3 _p17;
    unsigned int tmp2;
    p3 _p18;
    unsigned int tmp3;
    p3 _p19;
    unsigned int splitk;
    p3 _p20;
    unsigned int activation;
    p3 _p21;
    void *ptr_tmp3;
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
    TORCH_CHECK(m >= 16,
                "mi350 a8w8 blockscale asm only suuport m>=16 now!");
    TORCH_CHECK(k >=512,
                "mi350 a8w8 blockscale asm only suuport k>=512 now!");
    KernelArgs args;
    size_t arg_size = sizeof(args);

    args.ptr_X = (void *)XQ.data_ptr();
    args.ptr_W = (void *)WQ.data_ptr();
    args.ptr_XQ = (void *)x_scale.data_ptr();
    args.ptr_WQ = (void *)w_scale.data_ptr();
    args.ptr_C = (void *)out.data_ptr();
    args.K = k;
    args.N = n;
    args.M = m;
    args.eprt_cnt = 1;
    args.Xs = k * XQ.element_size();
    args.Ws = k * XQ.element_size();
    args.Cs = n * 2;
    args.splitk = 0;
    args.activation = 0;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(XQ));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    // printf("ptr_X: %p\n", args.ptr_X);
    // printf("ptr_GU: %p\n", args.ptr_GU);
    // printf("ptr_XQ: %p\n", args.ptr_XQ);
    // printf("ptr_GUQ: %p\n", args.ptr_GUQ);
    // printf("args.Xs: %d\n", args.Xs);
    // printf("args.token_cnt: %d\n", args.token_cnt);
    AiterAsmKernel<> *impl_ptr = nullptr;
    static AiterAsmKernel<{"f8_block_scale_mi350_x128", f8_block_scale_mi350_x128_co}> impl_kenrel_x128;
    static AiterAsmKernel<{"f8_block_scale_mi350_x32", f8_block_scale_mi350_x32_co}> impl_kenrel_x32;
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
