// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Skinny GEMM with BF16 activations and FP8 weights for decode on gfx950:
// out = (A @ B.T) * B_scale, with A [M, K] bf16, B [N, K] fp8 e4m3fn, one fp32
// scale per weight row in B_scale [N], and out [M, N] bf16, for M from 1 to 8.
//
// Every workgroup owns a few weight rows and all M tokens. The FP8 weights are
// converted to BF16, which is exact because every e4m3fn value is a BF16 value,
// and multiplied with the BF16 activations into fp32 sums. The sum is scaled by
// the row scale and rounded to BF16. The activations are never quantized. The
// kernel reads half the weight bytes of a BF16 GEMM, and at these M the weights
// are almost all of the memory traffic.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_a16w8gemm_configs.hpp"
#include <hip/hip_runtime.h>

// The kernel reads four 64-bit pointers packed back to back. Unlike most asm
// kernels in AITER, there is no padding between the arguments.
struct __attribute__((packed)) A16W8KernelArgs
{
    const void* a;
    const void* b;
    const void* b_scale;
    void* out;
};
static_assert(sizeof(A16W8KernelArgs) == 32, "the kernel reads a 32 byte argument buffer");

static const a16w8gemmConfig& get_a16w8gemm_config(const std::string& arch_id, int m, int n, int k)
{
    for(const auto& el : cfg_a16w8gemm)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.m == m && cfg.n == n && cfg.k == k)
            return cfg;
    }
    AITER_CHECK(false,
                "gemm_a16w8_asm: no kernel for arch ",
                arch_id,
                " with M=",
                m,
                " N=",
                n,
                " K=",
                k,
                ". The kernels cover gfx950 with M from 1 to 8 and (N, K) = (4608, 8192) or "
                "(8192, 2048).");
    __builtin_unreachable();
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(gemm_a16w8_asm,
                                    (aiter_tensor_t* A,       // [M, K] bf16
                                     aiter_tensor_t* B,       // [N, K] fp8 e4m3fn
                                     aiter_tensor_t* B_scale, // [N] or [N, 1] fp32
                                     aiter_tensor_t* out,     // [M, N] bf16
                                     hipStream_t stream),
                                    (A, B, B_scale, out, stream))
{
    const char* fn = "gemm_a16w8_asm";
    AITER_CHECK(A->dtype() == AITER_DTYPE_bf16,
                fn,
                ": A must be bf16, got ",
                AiterDtype_to_str(A->dtype()));
    AITER_CHECK(B->dtype() == AITER_DTYPE_fp8,
                fn,
                ": B must be fp8 e4m3fn, got ",
                AiterDtype_to_str(B->dtype()));
    AITER_CHECK(B_scale->dtype() == AITER_DTYPE_fp32,
                fn,
                ": B_scale must be float32, got ",
                AiterDtype_to_str(B_scale->dtype()));
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                fn,
                ": out must be bf16, got ",
                AiterDtype_to_str(out->dtype()));
    AITER_CHECK(
        A->dim() == 2 && B->dim() == 2 && out->dim() == 2, fn, ": A, B and out must be 2-D");

    const int m = A->size(0);
    const int k = A->size(1);
    const int n = B->size(0);
    if(m == 0)
        return;

    // The kernel addresses every tensor with fixed row strides, so the checks
    // below reject views that it would read or write in the wrong place.
    AITER_CHECK(A->stride(1) == 1 && A->stride(0) == k, fn, ": A must be contiguous");
    AITER_CHECK(B->size(1) == k && B->stride(1) == 1 && B->stride(0) == k,
                fn,
                ": B must be a contiguous [N, ",
                k,
                "] tensor");
    AITER_CHECK(B_scale->numel() == static_cast<size_t>(n) && B_scale->is_contiguous(),
                fn,
                ": B_scale must be a contiguous tensor of ",
                n,
                " row scales");
    AITER_CHECK(out->size(0) == m && out->size(1) == n && out->stride(1) == 1 &&
                    out->stride(0) == n,
                fn,
                ": out must be a contiguous [",
                m,
                ", ",
                n,
                "] tensor");
    const int dev = A->device_id;
    AITER_CHECK(B->device_id == dev && B_scale->device_id == dev && out->device_id == dev,
                fn,
                ": all tensors must be on the same GPU");

    const HipDeviceGuard device_guard(dev);
    const std::string arch_id = get_gpu_arch();
    const auto& cfg           = get_a16w8gemm_config(arch_id, m, n, k);

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();
    AiterAsmKernel* impl_ptr =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });

    A16W8KernelArgs args;
    args.a          = A->ptr;
    args.b          = B->ptr;
    args.b_scale    = B_scale->ptr;
    args.out        = out->ptr;
    size_t arg_size = sizeof(args);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             cfg.grid,    // gdx: N / rows per workgroup
                             1,           // gdy
                             1,           // gdz
                             cfg.threads, // bdx
                             1,           // bdy
                             1,           // bdz
                             stream});
}
