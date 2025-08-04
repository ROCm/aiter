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

    args.a_ptr = (void *)XQ.data_ptr();
    args.b_ptr = (void *)WQ.data_ptr();
    args.c_ptr = nullptr;
    args.sa_ptr = (void *)x_scale.data_ptr();
    args.sb_ptr = (void *)w_scale.data_ptr();
    args.d_ptr = nullptr;
    args.d_f16_ptr = (void *)out.data_ptr();

    args.num_tokens = m;
    args.intermediate_size = n;
    args.hidden_size = k;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(XQ));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AiterAsmKernel *impl_ptr = nullptr;
    static AiterAsmKernel impl_kenrel("flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32", "flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32.co");
    impl_ptr = &impl_kenrel;

    int gdx = (n + TileN - 1) / TileN;
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
