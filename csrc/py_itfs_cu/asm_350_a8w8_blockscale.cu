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
    constexpr int TileM = 128;
    constexpr int TileN = 256;
    constexpr int TileK = 128;

    int m = XQ.size(0);
    int n = out.size(1);
    int k = XQ.size(1);

    TORCH_CHECK(out.dtype() == torch::ScalarType::Half,
                "flatmm a8w8 blockscale asm only support Half output now!");
    TORCH_CHECK(n % TileN == 0 && k % TileK == 0, 
                "flatmm a8w8 blockscale asm only suuport 128x256x128 tile now!");

    KernelArgs args;
    size_t arg_size = sizeof(args);

    args.ptr_X = (void *)XQ.data_ptr();
    args.ptr_GU = (void *)WQ.data_ptr();
    args.ptr_XQ = (void *)x_scale.data_ptr();
    args.ptr_GUQ = (void *)w_scale.data_ptr();
    args.ptr_O = (void *)out.data_ptr();
    args.dim = k;
    args.hidden_dim = m;
    args.token_cnt = m;
    args.eprt_cnt = 1;
    args.Xs = k * XQ.element_size();
    args.GUs = k * XQ.element_size();
    args.Os = n * out.element_size();
    args.splitk = 0;
    args.activation = 0;
    const at::cuda::OptionalCUDAGuard device_guard(device_of(XQ));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel *impl_ptr = nullptr;
    static AiterAsmKernel impl_kenrel("flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32", "flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32.co");
    impl_ptr = &impl_kenrel;
    //  sz_stp = sz_sw = topk*batch + eprt*sub_X - topk; //max_length
    // sz_sep = (sz_stp + sub_X - 1)/sub_X; 
    //     int gdx = ((hidden_dim+sub_GU*2-1)/(sub_GU*2));
    // int gdy = sz_sep; 
    int gdx = (n + TileN*2 - 1) / (TileN*2);
    int gdy = (m + TileM - 1) / TileM;

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
