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
    void *ptr_C;
    p2 _p0;
    void *ptr_A;
    p2 _p1;
    void *ptr_B;
    p2 _p2;
    void *ptr_a_scale;
    p2 _p3;
    void *ptr_b_scale;
    p2 _p4;
    void *ptr_bias;
    p2 _p5;
    unsigned int m;
    p3 _p6;
    unsigned int n;
    p3 _p7;
    unsigned int k;
    p3 _p17;
    unsigned int lda;
    p3 _p8;
    unsigned int ldb;
    p3 _p9;
    unsigned int ldc;
    p3 _p10;
    unsigned int ks;
    p3 _p11;
    unsigned int scale_m;
    p3 _p12;
    unsigned int scale_n;
    p3 _p13;
    unsigned int scale_k;
    p3 _p14;
};

using namespace hip_fp8_impl;
torch::Tensor a8w8_blockscale_bpreshuffle_asm(
    torch::Tensor &A,      // [M, K]
    torch::Tensor &B,      // [N, K] -> [N/128, K*128]
    torch::Tensor &a_scale, // [M, K/128]
    torch::Tensor &b_scale, // [N/128, K/128]
    torch::Tensor &out,      // Out:[M, N] bf16
    torch::Tensor &bias     // [1, N]      fp32
)
{
    int TileM = 128;
    constexpr int TileN = 128;
    constexpr int TileK = 128;
    const unsigned int block_shape_m     = 1;
    const unsigned int block_shape_k     = 128;
    const unsigned int block_shape_n     = 128;


    int m = A.size(0);
    int n = B.size(0);
    int k = A.size(1);
    TORCH_CHECK(out.dtype() == torch::ScalarType::BFloat16,
                "mi300 a8w8 blockscale asm only support Half output now!");
    TORCH_CHECK(n % TileN == 0 && k % TileK == 0, 
                "mi300 a8w8 blockscale asm only suuport 128x128x128 tile now!");
    KernelArgs args;
    size_t arg_size = sizeof(args);

    args.ptr_A = (void *)A.data_ptr();
    args.ptr_B = (void *)B.data_ptr();
    args.ptr_a_scale = (void *)a_scale.data_ptr();
    args.ptr_b_scale = (void *)b_scale.data_ptr();
    args.ptr_C = (void *)out.data_ptr();
    args.k = k;
    args.n = n;
    args.m = m;
    args.lda = k;
    args.ldb = k;
    args.ldc = n * 2;
    args.ks = 0;
    args.scale_m = (m+block_shape_m-1) / block_shape_m;
    args.scale_n = (n+block_shape_n-1) / block_shape_n;
    args.scale_k = (k+block_shape_k-1) / block_shape_k;
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel *impl_ptr = nullptr;
    static AiterAsmKernel impl_kenrel_x128("f8_block_scale_mi300_x128", "f8_block_scale_mi300_x128.co");
    impl_ptr = &impl_kenrel_x128;
    int gdx = (n + TileN - 1) / (TileN);
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
