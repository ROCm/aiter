// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "aiter_hip_common.h"

struct __attribute__((packed)) KernelArgs
{
    void *ptr_O;
    p2 _p0;
    void *ptr_In;
    p2 _p1;
    void *ptr_Weight;
    p2 _p2;
    void *ptr_Bias;
    p2 _p3;
    float epsilon;
    p3 _p4;
    unsigned int M;
    p3 _p5;
    unsigned int N;
    p3 _p6;
    void *ptr_OutResidual;
    p2 _p7;
    void *ptr_InResidual;
    p2 _p8;
    void *ptr_OutYScale;
    p2 _p9;
    void *ptr_XScale;
    p2 _p10;
};

AITER_C_ITFS void layernorm2d_with_add_asm(aiter_tensor_t* out,          // [m ,n]
                              aiter_tensor_t* input,        // [m ,n]
                              aiter_tensor_t* residual_in,  // [m ,n]
                              aiter_tensor_t* residual_out, // [m ,n]
                              aiter_tensor_t* weight,       // [1 ,n]
                              aiter_tensor_t* bias,         // [1 ,n]
                              float epsilon,
                              hipStream_t stream)
{
    auto dtype = input->dtype();
    AITER_CHECK(dtype == AITER_DTYPE_bf16 ,
                __func__, " for now only support bf16 data type");
    AITER_CHECK(input->is_contiguous(),
                __func__, " for now only support input.is_contiguous()");

    KernelArgs args;
    int n = input->size(-1);
    int m = input->numel() / n;
    AITER_CHECK(m % 2 == 0,
                __func__, " for now only support m % 2 == 0");
    AITER_CHECK(n == 8192,
                __func__, " for now only support n == 8192");

    const HipDeviceGuard device_guard(input->device_id);

    size_t arg_size = sizeof(args);
    args.ptr_O = out->data_ptr();
    args.ptr_In = input->data_ptr();
    args.ptr_Weight = weight->data_ptr();
    args.ptr_Bias = bias->data_ptr();
    args.epsilon = epsilon;
    args.M = m;
    args.N = n;
    args.ptr_OutResidual = residual_out->data_ptr();
    args.ptr_InResidual = residual_in->data_ptr();

    int sub_M = 2;
    static AiterAsmKernel impl("layer_norm_kernel_func", "layer_norm.co");

    impl.launch_kernel({&args,
                        &arg_size,
                        ((m + sub_M - 1) / sub_M), // gdx
                        1,                         // gdy
                        1,                         // gdz
                        256,                       // bdx: 4 wv64
                        1,                         // bdy
                        1,                         // bdz
                        stream});
}

AITER_C_ITFS void layernorm2d_with_add_smoothquant_asm(aiter_tensor_t* out,          // [m ,n]
                                          aiter_tensor_t* input,        // [m ,n]
                                          aiter_tensor_t* residual_in,  // [m ,n]
                                          aiter_tensor_t* residual_out, // [m ,n]
                                          aiter_tensor_t* xscale,       // [1 ,n]
                                          aiter_tensor_t* yscale,       // [m ,1]
                                          aiter_tensor_t* weight,       // [1 ,n]
                                          aiter_tensor_t* bias,         // [1 ,n]
                                          float epsilon,
                                          hipStream_t stream)
{
    auto dtype = input->dtype();
    AITER_CHECK(dtype == AITER_DTYPE_bf16,
                __func__, " for now only support bf16 data type");
    AITER_CHECK(input->is_contiguous(),
                __func__, " for now only support input.is_contiguous()");

    KernelArgs args;
    int n = input->size(-1);
    int m = input->numel() / n;
    AITER_CHECK(m % 2 == 0,
                __func__, " for now only support m % 2 == 0");
    AITER_CHECK(n == 8192,
                __func__, " for now only support n == 8192");

    const HipDeviceGuard device_guard(input->device_id);

    size_t arg_size = sizeof(args);
    args.ptr_O = out->data_ptr();
    args.ptr_In = input->data_ptr();
    args.ptr_Weight = weight->data_ptr();
    args.ptr_Bias = bias->data_ptr();
    args.epsilon = epsilon;
    args.M = m;
    args.N = n;
    args.ptr_OutResidual = residual_out->data_ptr();
    args.ptr_InResidual = residual_in->data_ptr();
    args.ptr_OutYScale = yscale->data_ptr();
    args.ptr_XScale = xscale->data_ptr();

    int sub_M = 2;
    static AiterAsmKernel impl("layer_norm_qnt", "layer_norm_qnt.co");

    impl.launch_kernel({&args,
                        &arg_size,
                        ((m + sub_M - 1) / sub_M), // gdx
                        1,                         // gdy
                        1,                         // gdz
                        256,                       // bdx: 4 wv64
                        1,                         // bdy
                        1,                         // bdz
                        stream});
}