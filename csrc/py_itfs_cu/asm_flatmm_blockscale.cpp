// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "hip_float8.h"

struct __attribute__((packed)) KernelArgs
{
    const void* a_ptr;  // [m, k]
    const void* b_ptr;  // [n, k] -> [n/128, k*128]
    const void* c_ptr;  // 
    const void* sa_ptr; // [k/128, m]
    const void* sb_ptr; // [k/128, n/128]
    void* d_ptr;        // 
    void* d_f16_ptr;    // [m, n]
    void* dbg_int_ptr;
    void* dbg_fp8_ptr;
    void* dbg_f16_ptr;
    void* dbg_fp32_ptr;

    int hidden_size;       // K
    int intermediate_size; // N
    int num_tokens;        // M

    int num_experts;
    int topk;
    int stride_token;
};

using namespace hip_fp8_impl;
torch::Tensor flatmm_a8w8_blockscale_asm(
    torch::Tensor &A,       // [M, K]
    torch::Tensor &B,       // [N, K] -> [N/128, K*128]
    torch::Tensor &A_scale, // [K/128, M]
    torch::Tensor &B_scale, // [K/128, N/128]
    torch::Tensor &out      // Out:[M, N] fp16
)
{
    int m = A.size(0);
    int n = out.size(1);
    int k = A.size(1);

    KernelArgs args;
    size_t arg_size = sizeof(args);

    args.a_ptr = (void *)A.data_ptr();
    args.b_ptr = (void *)B.data_ptr();
    args.c_ptr = nullptr;
    args.sa_ptr = (void *)A_scale.data_ptr();
    args.sb_ptr = (void *)B_scale.data_ptr();
    args.d_ptr = nullptr;
    args.d_f16_ptr = (void *)out.data_ptr();

    args.num_tokens = m;
    args.intermediate_size = n;
    args.hidden_size = k;

    //const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel *impl_ptr = nullptr;
    static AiterAsmKernel impl_kenrel("flatmm_f8_uk_gfx9_128x128x128_1x4x1_16x16x32", "flatmm_f8_uk_gfx9_128x128x128_1x4x1_16x16x32.co");
    impl_ptr = &impl_kenrel;

    int gdx = n / 128;
    int gdy = m / 128;

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
