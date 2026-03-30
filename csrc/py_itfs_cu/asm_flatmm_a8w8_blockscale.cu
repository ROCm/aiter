// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "aiter_hip_common.h"

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

AITER_C_ITFS
void flatmm_a8w8_blockscale_asm(
    aiter_tensor_t *XQ,      // [M, K]
    aiter_tensor_t *WQ,      // [N, K] -> [N/128, K*128]
    aiter_tensor_t *x_scale, // [K/128, M]
    aiter_tensor_t *w_scale, // [K/128, N/128]
    aiter_tensor_t *out,     // Out:[M, N] fp16
    hipStream_t stream)
{
    constexpr int TileM = 128;
    constexpr int TileN = 256;
    constexpr int TileK = 128;

    int m = XQ->size(0);
    int n = out->size(1);
    int k = XQ->size(1);

    AITER_CHECK(out->dtype() == AITER_DTYPE_fp16,
                "flatmm a8w8 blockscale asm only support Half output now!");
    AITER_CHECK(n % TileN == 0 && k % TileK == 0,
                "flatmm a8w8 blockscale asm only support 128x256x128 tile now!");

    HipDeviceGuard device_guard(XQ->device_id);

    KernelArgs args;
    size_t arg_size = sizeof(args);

    args.a_ptr = XQ->data_ptr();
    args.b_ptr = WQ->data_ptr();
    args.c_ptr = nullptr;
    args.sa_ptr = x_scale->data_ptr();
    args.sb_ptr = w_scale->data_ptr();
    args.d_ptr = nullptr;
    args.d_f16_ptr = out->data_ptr();

    args.num_tokens = m;
    args.intermediate_size = n;
    args.hidden_size = k;

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
}
