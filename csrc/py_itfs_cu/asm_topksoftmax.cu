// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_T;
    p2 _p0;
    void* ptr_W;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    unsigned int batch;
    p3 _p4;
    unsigned int expert;
    p3 _p5;
    unsigned int topk;
    p3 _p6;
    unsigned int renormalize;
    p3 _p7;
    unsigned int out_stride;
    p3 _p8;
};

void topk_softmax_asm(torch::Tensor& topk_weights,         // [num_tokens, topk]
                      torch::Tensor& topk_indices,         // [num_tokens, topk]
                      torch::Tensor& token_expert_indices, // [num_tokens, topk]
                      torch::Tensor& gating_output,        // [num_tokens, num_experts]
                      bool need_renorm)
{
    const uint num_experts = gating_output.size(-1);
    const uint num_tokens  = gating_output.numel() / num_experts;
    const uint topk        = topk_weights.size(-1);
    const uint out_stride  = topk_weights.stride(0);
    const uint SUBM        = num_tokens < 10000 ? 4 : 12;

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_T      = (void*)topk_indices.data_ptr();
    args.ptr_W      = (void*)topk_weights.data_ptr();
    args.ptr_A      = (void*)gating_output.data_ptr();

    args.batch       = num_tokens;
    args.expert      = num_experts;
    args.topk        = topk;
    args.renormalize = need_renorm ? 1 : 0;
    args.out_stride  = out_stride * 4;

    AiterAsmKernel* impl_ptr = nullptr;
    if(num_experts == 128 && topk == 8)
    {
        if(SUBM == 4)
        {
            static AiterAsmKernel impl_topksoftmax_4x128x8("_ZN5aiter19topksoftmax_4x128x8E",
                                                           "/topksoftmax/topksoftmax_4x128x8.co");
            impl_ptr = &impl_topksoftmax_4x128x8;
        }
        else
        {
            static AiterAsmKernel impl_topksoftmax_12x128x8("_ZN5aiter20topksoftmax_12x128x8E",
                                                            "/topksoftmax/topksoftmax_12x128x8.co");
            impl_ptr = &impl_topksoftmax_12x128x8;
        }
    }
    else if(num_experts == 256 && topk == 8)
    {
        if(SUBM == 4)
        {
            static AiterAsmKernel impl_topksoftmax_4x256x8("_ZN5aiter19topksoftmax_4x256x8E",
                                                           "/topksoftmax/topksoftmax_4x256x8.co");
            impl_ptr = &impl_topksoftmax_4x256x8;
        }
        else
        {
            static AiterAsmKernel impl_topksoftmax_12x256x8("_ZN5aiter20topksoftmax_12x256x8E",
                                                            "/topksoftmax/topksoftmax_12x256x8.co");
            impl_ptr = &impl_topksoftmax_12x256x8;
        }
    }
    else if(num_experts == 128 && topk == 6)
    {
        if(SUBM == 4)
        {
            static AiterAsmKernel impl_topksoftmax_4x128x6("_ZN5aiter19topksoftmax_4x128x6E",
                                                           "/topksoftmax/topksoftmax_4x128x6.co");
            impl_ptr = &impl_topksoftmax_4x128x6;
        }
        else
        {
            static AiterAsmKernel impl_topksoftmax_12x128x6("_ZN5aiter20topksoftmax_12x128x6E",
                                                            "/topksoftmax/topksoftmax_12x128x6.co");
            impl_ptr = &impl_topksoftmax_12x128x6;
        }
    }
    else if(num_experts == 256 && topk == 6)
    {
        if(SUBM == 4)
        {
            static AiterAsmKernel impl_topksoftmax_4x256x6("_ZN5aiter19topksoftmax_4x256x6E",
                                                           "/topksoftmax/topksoftmax_4x256x6.co");
            impl_ptr = &impl_topksoftmax_4x256x6;
        }
        else
        {
            static AiterAsmKernel impl_topksoftmax_12x256x6("_ZN5aiter20topksoftmax_12x256x6E",
                                                            "/topksoftmax/topksoftmax_12x256x6.co");
            impl_ptr = &impl_topksoftmax_12x256x6;
        }
    }
    else
    {
        TORCH_CHECK(
            false,
            __func__,
            " only support num_experts/topk in [128/6, 128/8, 256/6, 256/8], but get num_experts: ",
            num_experts,
            " , topk: ",
            topk);
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uint gdx = (num_tokens + SUBM - 1) / SUBM;
    TORCH_CHECK(gdx >> 31 == 0, "num_tokens too large: ", num_tokens);
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             static_cast<int>(gdx), // gdx
                             1,                     // gdy
                             1,                     // gdz
                             256,                   // bdx: 4 wv64
                             1,                     // bdy
                             1,                     // bdz
                             stream});
}