// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_i8gemm_configs.hpp"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

// start to prepare the input and output buffer
struct __attribute__((packed)) KernelArgs
{
    void* ptr_c;
    p2 _p0;
    void* ptr_a;
    p2 _p1;
    void* ptr_b;
    p2 _p2;
    void* ptr_sa;
    p2 _p3;
    void* ptr_sb;
    p2 _p4;
    void* ptr_bias;
    p2 _p5;
    // float alpha;
    unsigned int m;
    p3 _p12;
    unsigned int n;
    p3 _p13;
    unsigned int k;
    p3 _p14;
    unsigned int lda;
    p3 _p15;
    unsigned int ldb;
    p3 _p16;
    unsigned int ldc;
    p3 _p17;
    unsigned int ks;
    p3 _p18;
};

static CFG* get_cfg(torch::Tensor& inp, torch::Tensor& out)
{
    if((inp.dtype() == torch::kInt8) && out.scalar_type() == at::ScalarType::BFloat16)

    {
        return &cfg_i8gemm_bf16_perTokenI8;
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
                                                  std::string arch_id,
                                                  std::optional<int> k_split,
                                                  std::optional<bool> bpreshuffle,
                                                  CFG* cfgs)
{
    k_split = k_split.value_or(0) ?: 1;
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu        = dev_prop.multiProcessorCount;
    uint32_t empty_cu      = num_cu;
    uint32_t tg_num        = 0;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0;
    // int k_split_en                 = (k_split.has_value() && k_split.value() != 0) ? 1 : 0;
    int k_split_en                 = 1;
    int bpreshuffle_en             = (bpreshuffle.has_value() && !bpreshuffle) ? 0 : 1;
    std::string selectedKernelName = "";
    int selectedsplitK             = 1;

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.bpreshuffle == bpreshuffle_en &&
           ((cfg.splitK == k_split_en) || !k_split.has_value()))
        {
            if((N % cfg.tile_n) == 0)
            {
                std::vector<int> splitK_list =
                    (k_split.has_value() && cfg.splitK)
                        ? std::vector<int>{k_split.value()}
                        : (cfg.splitK
                               ? std::vector<
                                     int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                               : std::vector<int>{1});

                for(auto& splitK : splitK_list)
                {
                    int tg_num_M         = (M + cfg.tile_m - 1) / cfg.tile_m;
                    int tg_num_N         = (N + cfg.tile_n - 1) / cfg.tile_n;
                    tg_num               = tg_num_M * tg_num_N * splitK;
                    uint32_t local_round = (tg_num + num_cu - 1) / num_cu;

                    float local_compute2mem_effi =
                        cfg.tile_m * cfg.tile_n / (cfg.tile_m + cfg.tile_n);

                    bool is_earlier_round        = (local_round < round);
                    bool is_same_round           = (local_round == round);
                    bool has_sufficient_empty_cu = (empty_cu > (local_round * num_cu - tg_num));
                    bool has_better_efficiency   = (local_compute2mem_effi > compute2mem_effi);

                    if(is_earlier_round ||
                       (is_same_round && (has_sufficient_empty_cu || has_better_efficiency)))
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
    return std::make_tuple(selectedKernelName, selectedsplitK);
}

torch::Tensor gemm_a8w8_asm(torch::Tensor& A,       // A:[M, K] i8
                            torch::Tensor& B,       // B:[N, K] i8 -> shuffle layout(32,16)
                            torch::Tensor& A_scale, // A_scale:[M, 1] f32
                            torch::Tensor& B_scale, // B_scale:[1, N] f32
                            torch::Tensor& out,     // Out:[M, N] bf16
                            std::string& kernelName,
                            torch::Tensor& bias, // bias:[1, N] f32
                            std::optional<bool> bpreshuffle = true,
                            std::optional<int> splitK       = std::nullopt)
{
    TORCH_CHECK(out.dtype() == torch::ScalarType::BFloat16,
                "GEMM A8W8 asm only support BFloat16 output now!");
    int Mdim     = A.size(0);
    int Ndim     = out.size(1);
    int Kdim     = A.size(1);
    int stride_a = A.stride(0);
    int stride_b = B.stride(0);
    int stride_c = out.stride(0);
    stride_c     = stride_c * sizeof(uint16_t);
    int ks       = splitK.value_or(0) ?: 1;

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_c      = (void*)out.data_ptr();
    args.ptr_a      = (void*)A.data_ptr();
    args.ptr_b      = (void*)B.data_ptr();
    args.ptr_sa     = (void*)A_scale.data_ptr();
    args.ptr_sb     = (void*)B_scale.data_ptr();
    args.ptr_bias   = (void*)bias.data_ptr();

    args.m   = Mdim;
    args.n   = Ndim;
    args.k   = Kdim;
    args.lda = stride_a;
    args.ldb = stride_b;
    args.ldc = stride_c;
    args.ks  = ks;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(A));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    CFG* config_map          = get_cfg(A, out);
    using DictKey            = std::tuple<int, int, int, std::optional<int>, std::optional<bool>>;
    struct SimpleHash
    {
        size_t operator()(const DictKey& key) const
        {
            const auto& [m, n, k, splitk, shuffle] = key;
            int splitk_key                         = splitk.has_value() ? splitk.value() : -1;
            bool shuffle_key                       = shuffle.has_value() ? shuffle.value() : false;
            return std::hash<int>()(m) ^ std::hash<int>()(n) ^ std::hash<int>()(k) ^
                   std::hash<int>()(splitk_key) ^ std::hash<bool>()(shuffle_key);
        }
    };
    static std::unordered_map<DictKey, std::tuple<std::string, int>, SimpleHash>
        heuristic_kernel_dict;

    if(config_map->empty())
    {
        TORCH_CHECK(false, __func__, " no kernel support a8w8 for this gpu arch");
    }
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    std::string arch_id = get_gpu_arch();
    kernelName          = kernelName.empty() ? "" : arch_id + kernelName;
    int selectedksplit  = splitK.value_or(0) ?: 1;
    if(kernelName.empty())
    {
        auto it = heuristic_kernel_dict.find(DictKey(Mdim, Ndim, Kdim, splitK, bpreshuffle));
        if(it != heuristic_kernel_dict.end())
        {
            auto res       = it->second;
            kernelName     = std::get<0>(res);
            selectedksplit = std::get<1>(res);
        }
        else
        {
            auto it =
                get_heuristic_kernel(Mdim, Ndim, Kdim, arch_id, splitK, bpreshuffle, config_map);

            kernelName     = std::get<0>(it);
            selectedksplit = std::get<1>(it);
            heuristic_kernel_dict[{Mdim, Ndim, Kdim, splitK, bpreshuffle}] =
                std::make_tuple(kernelName, selectedksplit);
        }
    }

    AiterAsmKernel* impl_ptr = nullptr;
    int SUBM                 = 0;
    int SUBN                 = 0;
    auto it                  = config_map->find(kernelName);
    int gdx                  = 0;
    int gdy                  = 0;
    int gdz                  = 0;
    int blockSizeX           = 256;
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        SUBM                = cfg.tile_m;
        SUBN                = cfg.tile_n;
        gdx                 = (Ndim / SUBN) * blockSizeX;
        gdy                 = (Mdim % SUBM == 0) ? Mdim / SUBM : Mdim / SUBM + 1;
        gdz                 = 1;

        if(cfg.splitK == 1 && selectedksplit > 0)
        {
            int k_per_split         = (Kdim + ks - 1) / selectedksplit;
            int k_per_split_aligned = ((k_per_split + 127) / 128) * 128;

            int actual_splitK = (Kdim + k_per_split_aligned - 1) / k_per_split_aligned;
            if(actual_splitK != selectedksplit)
            {
                printf("warning: change splitK form %d to %d to make sure every block deals with "
                       "128x k\n",
                       selectedksplit,
                       actual_splitK);
                selectedksplit = actual_splitK;
            }

            k_per_split         = (Kdim + selectedksplit - 1) / selectedksplit;
            k_per_split_aligned = ((k_per_split + 127) / 128) * 128;
            // printf("K info: _s_K=%d, _s_splitK=%d, _s_K_per_tg=%d, k_per_split_aligned=%d\n",
            //        Kdim,
            //        selectedksplit,
            //        k_per_split,
            //        k_per_split_aligned);
            args.ks = selectedksplit;
            // if(selectedksplit > 1)
            //     out.zero_();
        }
        gdx         = gdx * selectedksplit;
        auto result = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel " + kernelName);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx / blockSizeX, // gdx
                             gdy,              // gdy
                             gdz,              // gdz
                             256,              // bdx: 4 wv64
                             1,                // bdy
                             1,                // bdz
                             stream});
    return out;
}
