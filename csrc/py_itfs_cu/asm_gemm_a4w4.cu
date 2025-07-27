// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_f4gemm_configs.hpp"
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <hip/hip_runtime.h>
#include <torch/all.h>

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
    // p3 _p23;
};

static CFG* get_cfg(torch::Tensor& inp, torch::Tensor& out)
{

#if defined(__Float4_e2m1fn_x2)
    if((inp.dtype() == torch::kFloat4_e2m1fn_x2 || inp.dtype() == torch::kUInt8) &&
       out.scalar_type() == at::ScalarType::BFloat16)
#else
    if((inp.dtype() == torch::kUInt8) && out.scalar_type() == at::ScalarType::BFloat16)
#endif
    {
        return &cfg_f4gemm_bf16_per1x32Fp4;
    }
    else
    {
        TORCH_CHECK(false,
                    __func__,
                    " Unsupported input_type:",
                    inp.scalar_type(),
                    ", out_type:",
                    out.scalar_type());
    }
};

std::tuple<std::string, int> get_heuristic_kernel(int M,
                                                  int N,
                                                  int K,
                                                  std::optional<int> log2_k_split,
                                                  std::optional<bool> bpreshuffle,
                                                  CFG* cfgs)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu     = dev_prop.multiProcessorCount;
    uint32_t empty_cu   = num_cu;
    uint32_t tg_num     = 0;
    uint32_t round      = 0xffffffff;
    int log2_k_split_en = (log2_k_split.has_value() && log2_k_split.value() != 0) ? 1 : 0;
    int bpreshuffle_en  = (bpreshuffle.has_value() && !bpreshuffle) ? 0 : 1;
    std::string selectedKernelName = "";
    int selectedsplitK             = 1;

    for(const auto& el : *cfgs)
    {
        const auto& cfg = el.second;
        if(cfg.bpreshuffle == bpreshuffle_en &&
           ((cfg.splitK == log2_k_split_en) || !log2_k_split_en))
        {
            if((N % cfg.tile_N) == 0)
            {
                std::vector<int> splitK_list =
                    (log2_k_split.has_value() && cfg.splitK)
                        ? std::vector<int>{1 << log2_k_split.value()}
                        : (cfg.splitK ? std::vector<int>{2, 4, 8, 16} : std::vector<int>{1});

                for(auto& splitK : splitK_list)
                {
                    int tg_num_M         = (M + cfg.tile_M - 1) / cfg.tile_M;
                    int tg_num_N         = (N + cfg.tile_N - 1) / cfg.tile_N;
                    tg_num               = tg_num_M * tg_num_N * splitK;
                    uint32_t local_round = (tg_num + num_cu - 1) / num_cu;

                    if(local_round < round ||
                       (local_round == round && empty_cu > (local_round * num_cu - tg_num)))
                    {
                        round              = local_round;
                        empty_cu           = local_round * num_cu - tg_num;
                        selectedKernelName = el.first;
                        selectedsplitK     = splitK;
                    }
                }
            }
        }
    }

    TORCH_CHECK(selectedKernelName != "", __func__, ": cannot get heuristic kernel!");
    int log2_result = 0;
    while(selectedsplitK >>= 1)
        ++log2_result;
    return std::make_tuple(selectedKernelName, log2_result);
}

// A4W4 asm gemm kernel
// D=A*B*alpha+beta*C
torch::Tensor gemm_a4w4_asm(torch::Tensor& A,       // A:[M, K/2] f4x2
                            torch::Tensor& B,       // B:[N, K/2] f4x2
                            torch::Tensor& A_scale, // A_scale:[M, K/32] e8m0 paded
                            torch::Tensor& B_scale, // B_scale:[N, K/32] e8m0 paded
                            torch::Tensor& out,     // Out:[M, N] bf16
                            std::string& kernelName,
                            std::optional<torch::Tensor>& bias, // bias:[M, N] f32
                            std::optional<float> alpha      = 1.0,
                            std::optional<float> beta       = 0.0,
                            std::optional<bool> bpreshuffle = true,
                            std::optional<int> log2_k_split = std::nullopt)

{

    TORCH_CHECK(
        out.dtype() == torch::ScalarType::BFloat16, __func__, " only support BFloat16 output now!");
    int Mdim = A.size(0);
    int Ndim = B.size(0);
    int Kdim = A.size(1) * 2; // always fp4_x2F
    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_D      = (void*)out.data_ptr();
    args.ptr_C      = bias.has_value() ? (void*)bias.value().data_ptr() : nullptr;
    args.ptr_A      = (void*)A.data_ptr();
    args.ptr_B      = (void*)B.data_ptr();

    args.alpha          = alpha.value();
    args.beta           = beta.value();
    args.stride_C0      = out.stride(0);
    args.stride_A0      = A.stride(0) * 2; // always fp4_x2
    args.stride_B0      = B.stride(0) * 2; // always fp4_x2
    args.M              = Mdim;
    args.N              = Ndim;
    args.K              = Kdim;
    args.ptr_ScaleA     = (void*)A_scale.data_ptr();
    args.ptr_ScaleB     = (void*)B_scale.data_ptr();
    args.stride_ScaleA0 = A_scale.stride(0);
    args.stride_ScaleB0 = B_scale.stride(0);
    args.log2_k_split   = 0;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CFG* config_map           = get_cfg(A, out);
    using DictKey             = std::tuple<int, int, int, std::optional<int>, std::optional<bool>>;
    struct SimpleHash
    {
        size_t operator()(const DictKey& key) const
        {
            const auto& [m, n, k, log2, shuffle] = key;
            int log2_key                         = log2.has_value() ? log2.value() : -1;
            bool shuffle_key                     = shuffle.has_value() ? shuffle.value() : false;
            return std::hash<int>()(m) ^ std::hash<int>()(n) ^ std::hash<int>()(k) ^
                   std::hash<int>()(log2_key) ^ std::hash<bool>()(shuffle_key);
        }
    };
    static std::unordered_map<DictKey, std::tuple<std::string, int>, SimpleHash>
        heuristic_kernel_dict;

    if(config_map->empty())
    {
        TORCH_CHECK(false, __func__, " no kernel support a4w4 for this gpu arch");
    }

    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;

    int selectedksplit = log2_k_split.has_value() ? log2_k_split.value() : 0;
    if(kernelName.empty())
    {
        auto it = heuristic_kernel_dict.find(DictKey(Mdim, Ndim, Kdim, log2_k_split, bpreshuffle));
        if(it != heuristic_kernel_dict.end())
        {
            auto res       = it->second;
            kernelName     = std::get<0>(res);
            selectedksplit = std::get<1>(res);
        }
        else
        {
            auto it = get_heuristic_kernel(Mdim, Ndim, Kdim, log2_k_split, bpreshuffle, config_map);

            kernelName     = std::get<0>(it);
            selectedksplit = std::get<1>(it);
            heuristic_kernel_dict[{Mdim, Ndim, Kdim, log2_k_split, bpreshuffle}] =
                std::make_tuple(kernelName, selectedksplit);
        }
    }

    AiterAsmKernel* impl_ptr = nullptr;
    int SUBM                 = 0;
    int SUBN                 = 0;
    int gdz                  = 1;

    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.name.c_str();
        const char* co_name = cfg.co_name.c_str();
        SUBM                = cfg.tile_M;
        SUBN                = cfg.tile_N;

        if(cfg.splitK == 1)
        {
            args.log2_k_split = selectedksplit;
            int k_num         = 1 << args.log2_k_split;
            TORCH_CHECK(Kdim % k_num == 0, __func__, " Kdim % (1 << args.log2_k_split) != 0 !");
            int k_per_tg = Kdim / k_num;
            k_per_tg     = ((k_per_tg + 256 - 1) / 256) * 256;
            gdz          = (Kdim + k_per_tg - 1) / k_per_tg;
        }

        auto result = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel " + kernelName);

    int gdx = (Ndim + SUBN - 1) / SUBN;
    int gdy = (Mdim + SUBM - 1) / SUBM;

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx, // gdx
                             gdy, // gdy
                             gdz, // gdz
                             256, // bdx: 4 wv64
                             1,   // bdy
                             1,   // bdz
                             stream});
    return out;
}
