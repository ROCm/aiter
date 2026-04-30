// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 F4GEMM ASM dispatch (preload SGPR mode).
// Two entrypoints:
//   - mxfp4_gemm_asm: D[M,N] bf16 = A[M,K/2] mxfp4 * B[N,K/2] mxfp4 (e8m0 scales)
//   - nvfp4_gemm_asm: D[M,N] bf16 = A[M,K/2] nvfp4 * B[N,K/2] nvfp4 (e4m3 scales + GlobalScale)
//
// KernelArgs layout matches `args_preload` in poc_kl/mi400/f4gemm/f4gemm.cpp.
// MXFP4 packs 72B (drops trailing GlobalScaleA/B); NVFP4 packs 80B.
//
// First version: cluster launch is disabled; only single-CU launches via
// hipModuleLaunchKernel. CSV variants encoding cluster shapes are not exposed
// here yet.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_f4gemm_mi400_configs.hpp"
#include <cmath>
#include <memory>
#include <hip/hip_runtime.h>

constexpr int F4_INTYPE_MXFP4 = 7;
constexpr int F4_INTYPE_NVFP4 = 8;
constexpr int MXFP4_SCALE_BLOCK = 32;
constexpr int NVFP4_SCALE_BLOCK = 16;

// Legacy-mode KernelArgs (16B-per-arg, 368B total). Field order must match
// poc_kl/mi400/f4gemm/f4gemm.cpp:387-411 exactly.
struct _p3 { unsigned int _p0, _p1, _p2; };
struct _p2 { unsigned int _p0, _p1; };
struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;                  _p2 _pad0;      // arg 0
    void* ptr_C;                  _p2 _pad1;      // arg 1
    void* ptr_A;                  _p2 _pad2;      // arg 2
    void* ptr_B;                  _p2 _pad3;      // arg 3
    unsigned int strideD0;        _p3 _pad4;      // arg 4
    unsigned int strideD1;        _p3 _pad5;      // arg 5
    unsigned int strideC0;        _p3 _pad6;      // arg 6
    unsigned int strideC1;        _p3 _pad7;      // arg 7
    unsigned int strideA0;        _p3 _pad8;      // arg 8
    unsigned int strideA1;        _p3 _pad9;      // arg 9
    unsigned int strideB0;        _p3 _pad10;     // arg 10
    unsigned int strideB1;        _p3 _pad11;     // arg 11
    unsigned int M;               _p3 _pad12;     // arg 12
    unsigned int N;               _p3 _pad13;     // arg 13
    unsigned int K;               _p3 _pad14;     // arg 14
    void* ptr_ScaleA;             _p2 _pad15;     // arg 15
    void* ptr_ScaleB;             _p2 _pad16;     // arg 16
    unsigned int ScaleA_stride0;  _p3 _pad17;     // arg 17
    unsigned int ScaleA_stride1;  _p3 _pad18;     // arg 18
    unsigned int ScaleB_stride0;  _p3 _pad19;     // arg 19
    unsigned int ScaleB_stride1;  _p3 _pad20;     // arg 20
    float GlobalScaleA;           _p3 _pad21;     // arg 21
    float GlobalScaleB;           _p3 _pad22;     // arg 22
};

static std::tuple<std::string, int> get_heuristic_kernel(
    int M, int N, int K, std::string arch_id, int intype, int a_preshuffle, CFG* cfgs)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu        = dev_prop.multiProcessorCount;
    uint32_t empty_cu      = num_cu;
    uint32_t tg_num        = 0;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0f;
    std::string selectedKernelName = "";

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.intype != intype || cfg.a_preshuffle != a_preshuffle)
            continue;
        if((N % cfg.tile_n) != 0)
            continue;

        int tg_num_M         = (M + cfg.tile_m - 1) / cfg.tile_m;
        int tg_num_N         = (N + cfg.tile_n - 1) / cfg.tile_n;
        tg_num               = tg_num_M * tg_num_N;
        uint32_t local_round = (tg_num + num_cu - 1) / num_cu;

        float local_compute2mem_effi =
            (float)(cfg.tile_m * cfg.tile_n) / (cfg.tile_m + cfg.tile_n);

        bool is_earlier_round        = (local_round < round);
        bool is_same_round           = (local_round == round);
        bool has_sufficient_empty_cu = (empty_cu > (local_round * num_cu - tg_num));
        bool has_better_efficiency   = (local_compute2mem_effi > compute2mem_effi);

        if(is_earlier_round ||
           (is_same_round && (has_sufficient_empty_cu || has_better_efficiency)))
        {
            round              = local_round;
            empty_cu           = local_round * num_cu - tg_num;
            compute2mem_effi   = local_compute2mem_effi;
            selectedKernelName = el.first;
        }
    }

    AITER_CHECK(selectedKernelName != "",
                __func__,
                ": cannot get heuristic kernel for intype=",
                intype,
                ", a_preshuffle=",
                a_preshuffle,
                ", M=",
                M,
                ", N=",
                N,
                ", K=",
                K);
    return std::make_tuple(selectedKernelName, 1);
}

// Shared dispatch body. NVFP4 path passes real GlobalScale floats; MXFP4 passes
// 0.0f and a smaller arg_size that drops the trailing float pair.
static void f4gemm_mi400_launch(aiter_tensor_t* A,
                                aiter_tensor_t* B,
                                aiter_tensor_t* ScaleA,
                                aiter_tensor_t* ScaleB,
                                aiter_tensor_t* out,
                                const char*     kernelName,
                                int             intype,
                                int             a_preshuffle,
                                float           GlobalScaleA,
                                float           GlobalScaleB,
                                hipStream_t     stream)
{
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                __func__,
                " only supports BFloat16 output");
    AITER_CHECK(intype == F4_INTYPE_MXFP4 || intype == F4_INTYPE_NVFP4,
                __func__,
                " unsupported intype ",
                intype);
    AITER_CHECK(a_preshuffle == 0 || a_preshuffle == 1,
                __func__,
                " a_preshuffle must be 0 or 1");

    int Mdim = A->size(0);
    int Ndim = B->size(0);
    int Kdim = A->size(1) * 2; // packed fp4: stored dim = K/2 bytes

    int scale_block = (intype == F4_INTYPE_NVFP4) ? NVFP4_SCALE_BLOCK : MXFP4_SCALE_BLOCK;
    AITER_CHECK(Kdim % scale_block == 0,
                __func__,
                " K must be divisible by scale block size (",
                scale_block,
                ")");

    // Strides in bytes (matches poc_kl/mi400/f4gemm/f4gemm.cpp).
    unsigned int stride_a = static_cast<unsigned int>(Kdim / 2);     // fp4 packed
    unsigned int stride_b = static_cast<unsigned int>(Kdim / 2);     // fp4 packed
    unsigned int stride_d = static_cast<unsigned int>(Ndim) * 2;     // bf16
    unsigned int stride_sa = static_cast<unsigned int>(Kdim / scale_block);
    unsigned int stride_sb = static_cast<unsigned int>(Kdim / scale_block);

    KernelArgs args{};
    args.ptr_D           = out->ptr;
    args.ptr_C           = out->ptr;
    args.ptr_A           = A->ptr;
    args.ptr_B           = B->ptr;
    args.strideD0        = stride_d;
    args.strideC0        = stride_d;
    args.strideA0        = stride_a;
    args.strideB0        = stride_b;
    args.M               = Mdim;
    args.N               = Ndim;
    args.K               = Kdim;
    args.ptr_ScaleA      = ScaleA->ptr;
    args.ptr_ScaleB      = ScaleB->ptr;
    args.ScaleA_stride0  = stride_sa;
    args.ScaleB_stride0  = stride_sb;
    args.GlobalScaleA    = GlobalScaleA;
    args.GlobalScaleB    = GlobalScaleB;

    size_t arg_size = sizeof(KernelArgs);  // always 368B

    const HipDeviceGuard device_guard(A->device_id);

    static CFG* config_map = &cfg_f4gemm_mi400;
    AITER_CHECK(!config_map->empty(),
                __func__,
                " no kernel registered for f4gemm_mi400; check AITER_GPU_ARCHS=gfx1250");

    std::string arch_id = get_gpu_arch();
    std::string selectedName =
        (kernelName && kernelName[0] != '\0') ? (arch_id + kernelName) : "";

    using DictKey = std::tuple<int, int, int, int, int>; // M,N,K,intype,apre
    struct DictHash
    {
        size_t operator()(const DictKey& k) const
        {
            const auto& [m, n, kk, it, ap] = k;
            return std::hash<int>()(m) ^ std::hash<int>()(n) ^ std::hash<int>()(kk) ^
                   std::hash<int>()(it) ^ std::hash<int>()(ap);
        }
    };
    static SynchronizedCache<DictKey, std::string, DictHash> heuristic_kernel_dict;

    if(selectedName.empty())
    {
        selectedName = heuristic_kernel_dict.get_or_create(
            DictKey(Mdim, Ndim, Kdim, intype, a_preshuffle), [&]() {
                auto [name, _] = get_heuristic_kernel(
                    Mdim, Ndim, Kdim, arch_id, intype, a_preshuffle, config_map);
                return name;
            });
    }

    auto it = config_map->find(selectedName);
    AITER_CHECK(it != config_map->end(),
                __func__,
                " kernel not in cfg_f4gemm_mi400: ",
                selectedName);

    const auto& cfg     = it->second;
    AITER_CHECK(cfg.intype == intype && cfg.a_preshuffle == a_preshuffle,
                __func__,
                " selected kernel ",
                selectedName,
                " mismatches requested intype/a_preshuffle");

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        cfg.knl_name, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    int gdx = (Ndim + cfg.tile_n - 1) / cfg.tile_n;
    int gdy = (Mdim + cfg.tile_m - 1) / cfg.tile_m;
    int gdz = 1;
    int bdx = 128; // 4 wave * 32 thread on gfx1250

    impl_ptr->launch_kernel({&args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, stream});
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp4_gemm_asm,
    (aiter_tensor_t* A,        // A:[M, K/2] fp4x2 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,        // B:[N, K/2] fp4x2 (always preshuffled)
     aiter_tensor_t* ScaleA,   // ScaleA:[M, K/32] e8m0 (shuffled)
     aiter_tensor_t* ScaleB,   // ScaleB:[N, K/32] e8m0 (shuffled)
     aiter_tensor_t* out,      // Out:[M, N] bf16
     const char*     kernelName,
     int             a_preshuffle,
     hipStream_t     stream),
    (A, B, ScaleA, ScaleB, out, kernelName, a_preshuffle, stream))
{
    f4gemm_mi400_launch(A, B, ScaleA, ScaleB, out,
                        kernelName, F4_INTYPE_MXFP4, a_preshuffle,
                        0.0f, 0.0f, stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    nvfp4_gemm_asm,
    (aiter_tensor_t* A,        // A:[M, K/2] fp4x2 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,        // B:[N, K/2] fp4x2 (always preshuffled)
     aiter_tensor_t* ScaleA,   // ScaleA:[M, K/32] e4m3 (shuffled)
     aiter_tensor_t* ScaleB,   // ScaleB:[N, K/32] e4m3 (shuffled)
     float           GlobalScaleA,
     float           GlobalScaleB,
     aiter_tensor_t* out,      // Out:[M, N] bf16
     const char*     kernelName,
     int             a_preshuffle,
     hipStream_t     stream),
    (A, B, ScaleA, ScaleB, GlobalScaleA, GlobalScaleB,
     out, kernelName, a_preshuffle, stream))
{
    f4gemm_mi400_launch(A, B, ScaleA, ScaleB, out,
                        kernelName, F4_INTYPE_NVFP4, a_preshuffle,
                        GlobalScaleA, GlobalScaleB, stream);
}
