// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_mxfp8_batch_gemm_configs.hpp"
#include <cmath>
#include <memory>
#include <optional>
#include <hip/hip_runtime.h>

// KernelArgs layout matches mxfp8fp4gemm.cpp reference:
// D(out), A, B(preshuffled), ScaleA(e8m0), ScaleB(e8m0),
// stride_C, stride_A, stride_B, ScaleA_K, ScaleB_K, M, N, K
struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;           // Out:[B, M, N] bf16
    p2 _pad0;
    void* ptr_A;           // A:[B, M, K] fp8 (preshuffled: m/2, k/128, 2, 128)
    p2 _pad1;
    void* ptr_B;           // B:[B, N, K] fp8 (preshuffled: n/16, k/16, 16, 16)
    p2 _pad2;
    void* ptr_ScaleA;      // ScaleA:[B, M, K/32] e8m0 (shuffled: m/32, k/4, 32, 4)
    p2 _pad3;
    void* ptr_ScaleB;      // ScaleB:[B, N, K/32] e8m0 (shuffled: n/32, k/4, 32, 4)
    p2 _pad4;
    unsigned int stride_C; // C stride in bytes
    p3 _pad5;
    unsigned int stride_A; // A stride (K for fp8, K/2 for fp4)
    p3 _pad6;
    unsigned int stride_B; // B stride (K for fp8, K/2 for fp4)
    p3 _pad7;
    unsigned int ScaleA_K; // K / SCALE_BLOCK_SIZE (= K/32)
    p3 _pad8;
    unsigned int ScaleB_K; // K / SCALE_BLOCK_SIZE (= K/32)
    p3 _pad9;
    unsigned int M;
    p3 _pad10;
    unsigned int N;
    p3 _pad11;
    unsigned int K;
    p3 _pad12;
};

static std::tuple<std::string, int> get_heuristic_kernel(
    int M, int N, int K, std::string arch_id, CFG* cfgs)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu        = dev_prop.multiProcessorCount;
    uint32_t empty_cu      = num_cu;
    uint32_t tg_num        = 0;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0;
    std::string selectedKernelName = "";

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if((N % cfg.tile_n) == 0)
        {
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
    }

    AITER_CHECK(selectedKernelName != "", __func__, ": cannot get heuristic kernel!");
    return std::make_tuple(selectedKernelName, 1);
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp8_batch_gemm_asm,
    (
    aiter_tensor_t* A,        // A:[B, M, K] fp8 (preshuffled)
    aiter_tensor_t* B,        // B:[B, N, K] fp8 (preshuffled)
    aiter_tensor_t* ScaleA,   // ScaleA:[B, M, K/32] uint8 e8m0 (shuffled)
    aiter_tensor_t* ScaleB,   // ScaleB:[B, N, K/32] uint8 e8m0 (shuffled)
    aiter_tensor_t* out,      // Out:[B, M, N] bf16
    const char*  kernelName,
    hipStream_t  stream),
    (A, B, ScaleA, ScaleB, out, kernelName, stream))
{
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                "MXFP8 batched GEMM ASM only supports BFloat16 output now!");

    int batch_size = A->size(0);
    int Mdim       = A->size(1);
    int Kdim       = A->size(2);
    int Ndim       = out->size(2);

    // Strides in bytes — same computation as mxfp8fp4gemm.cpp reference
    // A is fp8: stride = K (1 byte per element)
    // B is fp8: stride = K (1 byte per element)
    // C is bf16: stride = N * sizeof(bf16)
    unsigned int stride_a = static_cast<unsigned int>(Kdim);      // fp8: 1 byte/elem
    unsigned int stride_b = static_cast<unsigned int>(Kdim);      // fp8: 1 byte/elem
    unsigned int stride_c = static_cast<unsigned int>(Ndim) * 2;  // bf16: 2 bytes/elem

    constexpr int SCALE_BLOCK_SIZE = 32;
    unsigned int ScaleA_K = Kdim / SCALE_BLOCK_SIZE;
    unsigned int ScaleB_K = Kdim / SCALE_BLOCK_SIZE;

    KernelArgs args;
    size_t arg_size   = sizeof(args);
    args.ptr_D        = out->ptr;
    args.ptr_A        = A->ptr;
    args.ptr_B        = B->ptr;
    args.ptr_ScaleA   = ScaleA->ptr;
    args.ptr_ScaleB   = ScaleB->ptr;
    args.stride_C     = stride_c;
    args.stride_A     = stride_a;
    args.stride_B     = stride_b;
    args.ScaleA_K     = ScaleA_K;
    args.ScaleB_K     = ScaleB_K;
    args.M            = Mdim;
    args.N            = Ndim;
    args.K            = Kdim;

    const HipDeviceGuard device_guard(A->device_id);

    static CFG* config_map = &cfg_mxfp8_batch_gemm;
    if(config_map->empty())
    {
        AITER_CHECK(false, __func__, " no kernel support mxfp8_batch_gemm for this gpu arch");
    }

    // Kernel selection: user-specified or heuristic
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    std::string arch_id = get_gpu_arch();
    std::string selectedName = (kernelName && kernelName[0] != '\0')
                                   ? arch_id + kernelName
                                   : "";

    using DictKey = std::tuple<int, int, int>;
    static std::unordered_map<DictKey, std::string,
        decltype([](const DictKey& k) {
            return std::hash<int>()(std::get<0>(k)) ^
                   std::hash<int>()(std::get<1>(k)) ^
                   std::hash<int>()(std::get<2>(k));
        })> heuristic_kernel_dict;

    if(selectedName.empty())
    {
        auto it = heuristic_kernel_dict.find({Mdim, Ndim, Kdim});
        if(it != heuristic_kernel_dict.end())
        {
            selectedName = it->second;
        }
        else
        {
            auto [name, _] = get_heuristic_kernel(Mdim, Ndim, Kdim, arch_id, config_map);
            selectedName = name;
            heuristic_kernel_dict[{Mdim, Ndim, Kdim}] = selectedName;
        }
    }

    // Look up config and launch
    AiterAsmKernel* impl_ptr = nullptr;
    int SUBM = 0;
    int SUBN = 0;
    auto it  = config_map->find(selectedName);
    AITER_CHECK(it != config_map->end(), __func__, " not find kernel ", selectedName);

    const auto& cfg     = it->second;
    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();
    SUBM                = cfg.tile_m;
    SUBN                = cfg.tile_n;

    // Grid dimensions match mxfp8fp4gemm.cpp reference:
    // gdx = (N + SUBN - 1) / SUBN
    // gdy = (M + SUBM - 1) / SUBM
    // gdz = batch_size
    int gdx = (Ndim + SUBN - 1) / SUBN;
    int gdy = (Mdim + SUBM - 1) / SUBM;
    int gdz = batch_size;
    int blockSizeX = 128;  // 4 waves * 32 threads

    auto result = impl_ptr_map.emplace(name, nullptr);
    if(result.second)
    {
        result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
    }
    impl_ptr = result.first->second.get();

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,
                             gdy,
                             gdz,
                             blockSizeX,
                             1,
                             1,
                             stream});
}
