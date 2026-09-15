// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

// Must follow aiter_tensor.h, which provides the HIP error-bridge dependencies.
#include "aiter_ctypes_error.h"
#include "asm_f6gemm_configs.hpp"
#include <cmath>
#include <hip/hip_runtime.h>
#include <memory>

namespace {
constexpr int kTileSize             = 256;
constexpr int kKTileSize            = 128;
constexpr int kScaleGroupSize       = 32;
constexpr int kKPaddingTiles        = 2;
constexpr size_t kPackedTileBytes   = 24576;
constexpr size_t kScaleTileBytes    = 1024;
constexpr char kPackLayout[]        = "mxfp6_c0c1_256_padk2";
constexpr char kDefaultKernelName[] = "f6gemm_dmabig_kernel_func";

// KernelArgs layout is identical to the a4w4 asm gemm ABI; the mxfp6 dmabig
// kernel was assembled against the same kernarg struct (0x180 bytes). Fields
// the fp6 kernel does not consume (ptr_C, beta, A/B strides, k-split) are left
// zeroed and ignored by the kernel.
struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;
    p2 _p0;
    void* ptr_C;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    void* ptr_B;
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
    void* ptr_ScaleA;
    p2 _p17;
    void* ptr_ScaleB;
    p2 _p18;
    unsigned int stride_ScaleA0;
    p3 _p19;
    unsigned int stride_ScaleA1;
    p3 _p20;
    unsigned int stride_ScaleB0;
    p3 _p21;
    unsigned int stride_ScaleB1;
    p3 _p22;
    int log2_k_split;
    p3 _p23;
};
static_assert(sizeof(KernelArgs) == 0x180, "a6w6 KernelArgs must be 0x180 bytes");

CFG* get_cfg(AiterDtype inp_dtype, AiterDtype out_dtype)
{
    if((inp_dtype == AITER_DTYPE_u8) && out_dtype == AITER_DTYPE_bf16)
    {
        return &cfg_f6gemm_bf16_per1x32Fp6;
    }
    else
    {
        AITER_CHECK(false,
                    __func__,
                    " Unsupported input_type:",
                    AiterDtype_to_str(inp_dtype),
                    ", out_type:",
                    AiterDtype_to_str(out_dtype));
        return nullptr;
    }
}
} // namespace

// A6W6 (mxfp6, E2M3, per-1x32 blockscale) asm gemm kernel
// D = A * B * alpha
AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    gemm_a6w6_asm,
    (aiter_tensor_t * A,      // A: packed mxfp6 blob (pack_big layout)
     aiter_tensor_t* B,       // B: packed mxfp6 blob (pack_big layout)
     aiter_tensor_t* A_scale, // A_scale: packed e8m0 blob (pack_scale layout)
     aiter_tensor_t* B_scale, // B_scale: packed e8m0 blob (pack_scale layout)
     aiter_tensor_t* out,     // Out:[M, N] bf16
     int K,                   // padded contraction dim consumed by the packed layout
     const char* kernelName,
     float alpha,
     hipStream_t stream),
    (A, B, A_scale, B_scale, out, K, kernelName, alpha, stream))
{
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16, __func__, " only support BFloat16 output now!");
    AITER_CHECK(out->dim() == 2 && out->is_contiguous(),
                __func__,
                " output must be a contiguous 2D tensor");
    AITER_CHECK(A->is_contiguous() && B->is_contiguous() && A_scale->is_contiguous() &&
                    B_scale->is_contiguous(),
                __func__,
                " packed operands and scales must be contiguous");
    AITER_CHECK(A->device_id == B->device_id && A->device_id == A_scale->device_id &&
                    A->device_id == B_scale->device_id && A->device_id == out->device_id,
                __func__,
                " all tensors must be on the same GPU");
    int Mdim = out->size(0);
    int Ndim = out->size(1);
    int Kdim = K;
    AITER_CHECK(
        Mdim > 0 && (Mdim % kTileSize) == 0, __func__, " M must be a positive multiple of 256!");
    AITER_CHECK(
        Ndim > 0 && (Ndim % kTileSize) == 0, __func__, " N must be a positive multiple of 256!");
    AITER_CHECK(
        Kdim > 0 && (Kdim % kKTileSize) == 0, __func__, " K must be a positive multiple of 128!");
    AITER_CHECK(A->dtype() == AITER_DTYPE_u8 && B->dtype() == AITER_DTYPE_u8 &&
                    A_scale->dtype() == AITER_DTYPE_u8 && B_scale->dtype() == AITER_DTYPE_u8,
                __func__,
                " packed operands and scales must use uint8");

    const size_t nk_pad     = static_cast<size_t>(Kdim / kKTileSize + kKPaddingTiles);
    const size_t expected_A = static_cast<size_t>(Mdim / kTileSize) * nk_pad * kPackedTileBytes;
    const size_t expected_B = static_cast<size_t>(Ndim / kTileSize) * nk_pad * kPackedTileBytes;
    const size_t expected_scaleA = static_cast<size_t>(Mdim / kTileSize) * nk_pad * kScaleTileBytes;
    const size_t expected_scaleB = static_cast<size_t>(Ndim / kTileSize) * nk_pad * kScaleTileBytes;
    AITER_CHECK(A->numel() == expected_A && B->numel() == expected_B,
                __func__,
                " packed operand buffer sizes must exactly match the launch dimensions");
    AITER_CHECK(A_scale->numel() == expected_scaleA && B_scale->numel() == expected_scaleB,
                __func__,
                " packed scale buffer sizes must exactly match the launch dimensions");

    KernelArgs args{};
    size_t arg_size     = sizeof(args);
    args.ptr_D          = out->ptr;
    args.ptr_C          = nullptr;
    args.ptr_A          = A->ptr;
    args.ptr_B          = B->ptr;
    args.alpha          = alpha;
    args.beta           = 0.0f;
    args.stride_D0      = out->stride(0);
    args.M              = Mdim;
    args.N              = Ndim;
    args.K              = Kdim;
    args.ptr_ScaleA     = A_scale->ptr;
    args.ptr_ScaleB     = B_scale->ptr;
    args.stride_ScaleA0 = Kdim / kScaleGroupSize;
    args.stride_ScaleB0 = Kdim / kScaleGroupSize;
    args.log2_k_split   = 0;

    const HipDeviceGuard device_guard(A->device_id);

    CFG* config_map = get_cfg(A->dtype(), out->dtype());
    AITER_CHECK(!config_map->empty(), __func__, " no kernel support a6w6 for this gpu arch");

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    std::string arch_id = get_gpu_arch();
    std::string kname   = (kernelName && kernelName[0] != 0) ? (arch_id + kernelName)
                                                             : (arch_id + kDefaultKernelName);

    AiterAsmKernel* impl_ptr = nullptr;
    int SUBM                 = 0;
    int SUBN                 = 0;
    int gdz                  = 1;
    int bdx                  = 0;

    auto it = config_map->find(kname);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        SUBM                = cfg.tile_M;
        SUBN                = cfg.tile_N;
        bdx                 = cfg.block_size;
        AITER_CHECK(SUBM > 0 && SUBN > 0, __func__, " invalid a6w6 tile dimensions");
        AITER_CHECK(bdx > 0, __func__, " invalid a6w6 workgroup size");
        AITER_CHECK(cfg.splitK == 0, __func__, " a6w6 split-K is not supported");
        AITER_CHECK(cfg.pack_layout == kPackLayout,
                    __func__,
                    " unsupported a6w6 packed layout " + cfg.pack_layout);
        if(cfg.swizzle_max_K > 0 && Kdim <= cfg.swizzle_max_K)
        {
            AITER_CHECK(Mdim <= cfg.swizzle_max_M,
                        __func__,
                        " kernel " + cfg.knl_name + " exceeds its M swizzle bound");
            AITER_CHECK(Ndim <= cfg.swizzle_max_N,
                        __func__,
                        " kernel " + cfg.knl_name + " exceeds its N swizzle bound");
        }

        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
    }
    else
        AITER_CHECK(false, __func__, " not find kernel " + kname);

    int gdx = (Ndim + SUBN - 1) / SUBN;
    int gdy = (Mdim + SUBM - 1) / SUBM;

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx, // gdx
                             gdy, // gdy
                             gdz, // gdz
                             bdx, // bdx
                             1,   // bdy
                             1,   // bdz
                             stream});
}
