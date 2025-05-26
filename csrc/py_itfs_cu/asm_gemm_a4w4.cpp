// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"

struct __attribute__((packed)) KernelArgs
{
    void *ptr_D;
    p2 _p0;
    void *ptr_C;
    p2 _p1;
    void *ptr_A;
    p2 _p2;
    void *ptr_B;
    p2 _p3;
    float alpha;
    p3 _p4;
    float beta;
    p3 _p5;

    unsigned int stride_D0;
    p3 _p6;
    unsigned int stride_D1;
    p3 _p7;
    unsigned int stride_C0;
    p3 _p8;
    unsigned int stride_C1;
    p3 _p9;
    unsigned int stride_A0;
    p3 _p10;
    unsigned int stride_A1;
    p3 _p11;
    unsigned int stride_B0;
    p3 _p12;
    unsigned int stride_B1;
    p3 _p13;
    unsigned int M;
    p3 _p14;
    unsigned int N;
    p3 _p15;
    unsigned int K;
    p3 _p16;
    void *ptr_ScaleA;
    p2 _p17;
    void *ptr_ScaleB;
    p2 _p18;
    unsigned int stride_ScaleA0;
    p3 _p19;
    unsigned int stride_ScaleA1;
    p3 _p20;
    unsigned int stride_ScaleB0;
    p3 _p21;
    unsigned int stride_ScaleB1;
    p3 _p22;
};

// A4W4 asm gemm kernel
// D=A*B*alpha+beta*C
torch::Tensor gemm_a4w4_asm(torch::Tensor &A,       // A:[M, K/2] f4x2
                            torch::Tensor &B,       // B:[N, K/2] f4x2
                            torch::Tensor &A_scale, // A_scale:[M, K/32] e8m0 paded
                            torch::Tensor &B_scale, // B_scale:[N, K/32] e8m0 paded
                            torch::Tensor &out,     // Out:[M, N] bf16
                            torch::Tensor &bias,    // bias:[M, N] f32
                            std::optional<float> alpha = 1.0,
                            std::optional<float> beta = 0.0)
{
    TORCH_CHECK(out.dtype() == torch::ScalarType::BFloat16,
                __func__, " only support BFloat16 output now!");
    int Mdim = A.size(0);
    int Ndim = B.size(0);
    int Kdim = A.size(1) * 2; // always fp4_x2

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_D = (void *)out.data_ptr();
    args.ptr_C = (void *)bias.data_ptr();
    args.ptr_A = (void *)A.data_ptr();
    args.ptr_B = (void *)B.data_ptr();

    args.alpha = alpha.value();
    args.beta = beta.value();
    args.stride_C0 = out.stride(0);
    args.stride_A0 = A.stride(0) * 2; // always fp4_x2
    args.stride_B0 = B.stride(0) * 2; // always fp4_x2
    args.M = Mdim;
    args.N = Ndim;
    args.K = Kdim;

    args.ptr_ScaleA = (void *)A_scale.data_ptr();
    args.ptr_ScaleB = (void *)B_scale.data_ptr();
    args.stride_ScaleA0 = A_scale.stride(0);
    args.stride_ScaleB0 = B_scale.stride(0);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // TODO should get from kernel
    int SUBM = 256;
    int SUBN = 256;
    static AiterAsmKernel noSplitK_impl("_ZN5aiter33f4gemm_bf16_per1x32Fp4_tn_256x256E", "f4gemm/f4gemm_bf16_per1x32Fp4_tn_256x256.co");
    AiterAsmKernel *impl_ptr = &noSplitK_impl;
    // if (ks > 0)
    //     impl_ptr = &splitK_impl;

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             (Ndim + SUBN - 1) / SUBN, // gdx
                             (Mdim + SUBM - 1) / SUBM, // gdy
                             1,                        // gdz
                             256,                      // bdx: 4 wv64
                             1,                        // bdy
                             1,                        // bdz
                             stream});
    return out;
}
