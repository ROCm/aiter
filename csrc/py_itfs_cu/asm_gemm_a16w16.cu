// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_bf16gemm_configs.hpp"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
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
    unsigned int splitk;
    p3 _p17;
    unsigned int is_out_b16;
    p3 _p18;
    void* ptr_Bias;
    p2 _p19;
    unsigned int add_bias;
    p3 _p20;
    void* ptr_semaphore;
    p2 _p21;
};

std::tuple<std::string, int>
get_heuristic_kernel(int M,
                     int N,
                     int K,
                     CFG* cfgs,
                     std::string arch_id,
                     bool bpreshuffle,
                     int add_bias,
                     std::optional<int> splitk             = std::nullopt,
                     std::optional<std::string> kernelName = std::nullopt)
{
    TORCH_CHECK(K % 64 == 0, __func__, " Kdim must be divisible by 64 !"); // load min size is 128b
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu = dev_prop.multiProcessorCount;

    uint32_t empty_cu      = num_cu;
    uint32_t pure_tg_num   = 0;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0;
    int oob                = M;

    std::string selectedKernelName = "";
    int selectedsplitK             = 1;

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(kernelName.has_value() && el.first != (arch_id + kernelName.value()))
            continue;
        // check specified kernel name
        if(kernelName.has_value())
        {
            TORCH_CHECK(N % cfg.tileN == 0 && cfg.bPreshuffle == (bpreshuffle ? 1 : 0) &&
                            (add_bias == 0 || cfg.bias == 1),
                        __func__,
                        " The specified kernel name ",
                        el.first,
                        " cannot support the input shape (N=",
                        N,
                        ", tileN=",
                        cfg.tileN,
                        ") or bias/preshuffle setting (preshuffle=",
                        bpreshuffle,
                        ", bias=",
                        add_bias,
                        ").");
            selectedKernelName = el.first;
            if(splitk.has_value())
            {
                selectedsplitK = splitk.value();
                selectedsplitK = std::min({selectedsplitK, 16, static_cast<int>(K / cfg.subK)});
                TORCH_CHECK((selectedsplitK > 1 && cfg.splitK == 1) ||
                                (selectedsplitK <= 1 && cfg.splitK == 0),
                            __func__,
                            " The specified splitK ",
                            selectedsplitK,
                            " cannot be supported by the specified kernel or Kdim",
                            el.first,
                            ".");
                break;
            }
        }
        // auto select splitk or kernel
        if(N % cfg.tileN == 0 && cfg.bPreshuffle == (bpreshuffle ? 1 : 0) &&
           (add_bias == 0 || cfg.bias == 1))
        {
            // 1. select splitk
            int split_K = 1;
            pure_tg_num = ((M + cfg.tileM - 1) / cfg.tileM) * (N / cfg.tileN);
            if(cfg.splitK == 1 && K / cfg.subK >= 2) // kernel and Kdim support splitk
            {
                TORCH_CHECK(cfg.subK > 0,
                            __func__,
                            " cfg.subK must be greater than 0 to avoid division by zero.");
                int max_splitk = std::min(std::min(static_cast<int>(num_cu / pure_tg_num), 16),
                                          static_cast<int>(K / cfg.subK));
                split_K =
                    std::max(2, max_splitk); // if kernel support splitk, set splitk to 2 at least.
            }
            // 2. better or not
            uint32_t tg_num      = pure_tg_num * split_K;
            uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
            float local_compute2mem_effi =
                static_cast<float>(cfg.tileM * cfg.tileN) / (cfg.tileM + cfg.tileN);
            bool is_earlier_round        = (local_round < round);
            bool is_same_round           = (local_round == round);
            bool has_sufficient_empty_cu = (empty_cu > (local_round * num_cu - tg_num));
            bool has_same_empty_cu       = (empty_cu == (local_round * num_cu - tg_num));
            bool has_better_efficiency   = (local_compute2mem_effi > compute2mem_effi);
            bool less_oob = (M % cfg.tileM == 0) ? (oob > 0) : (cfg.tileM - M % cfg.tileM < oob);
            bool has_same_oob = (cfg.tileM - (M % cfg.tileM)) == oob;

            if(is_earlier_round || (is_same_round && (has_sufficient_empty_cu || less_oob)) ||
               (is_same_round && has_same_empty_cu && has_same_oob && has_better_efficiency))
            {
                round              = local_round;
                empty_cu           = local_round * num_cu - tg_num;
                compute2mem_effi   = local_compute2mem_effi;
                oob                = (M % cfg.tileM == 0) ? 0 : cfg.tileM - (M % cfg.tileM);
                selectedKernelName = el.first;
                selectedsplitK     = split_K;
            }
        }
    }
    TORCH_CHECK(
        selectedKernelName != "", __func__, " not find kernel for bf16gemm~ " + selectedKernelName);
    return std::make_tuple(selectedKernelName, selectedsplitK);
}

AiterAsmKernel* get_or_load_kernel(const std::string& selectedKernelName,
                                   CFG* config_map,
                                   unsigned int& SUBM,
                                   unsigned int& SUBN)
{
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;

    auto it_kl = config_map->find(selectedKernelName);
    TORCH_CHECK(it_kl != config_map->end(), __func__, " not find kernel~ " + selectedKernelName);

    const auto& cfg     = it_kl->second;
    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();
    SUBM                = cfg.tileM;
    SUBN                = cfg.tileN;

    auto result = impl_ptr_map.emplace(name, nullptr);
    if(result.second)
        result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);

    return result.first->second.get();
}

torch::Tensor gemm_a16w16_asm(torch::Tensor& A,
                              torch::Tensor& B,
                              torch::Tensor& out,
                              torch::Tensor& semaphore,
                              std::optional<torch::Tensor> bias,
                              std::optional<int> splitK,
                              std::optional<std::string> kernelName,
                              bool bpreshuffle = false)
{
    TORCH_CHECK(out.dtype() == torch::ScalarType::Float ||
                    out.dtype() == torch::ScalarType::BFloat16,
                "GEMM A16W16 asm only support Float32 or Bf16 output now!");

    std::string arch_id = get_gpu_arch();
    int Mdim            = A.size(0);
    int Ndim            = B.size(0);
    int Kdim            = A.size(1);

    unsigned int SUBM = 32;
    unsigned int SUBN = 64;

    KernelArgs args = {};
    args.ptr_D      = (void*)out.data_ptr();
    args.ptr_C      = nullptr;
    args.ptr_A      = (void*)A.data_ptr();
    args.ptr_B      = (void*)B.data_ptr();
    args.ptr_Bias   = bias.has_value() ? (void*)bias.value().data_ptr() : nullptr;
    args.alpha      = 1.0f;
    args.beta       = 0.0f;
    args.stride_A0  = A.stride(0) * A.element_size();
    args.stride_B0  = B.stride(0) * B.element_size();
    args.stride_C0 = args.stride_D0 = Ndim * out.element_size();
    args.M                          = Mdim;
    args.N                          = Ndim;
    args.K                          = Kdim;
    args.is_out_b16                 = (out.dtype() == torch::ScalarType::BFloat16) ? 1 : 0;
    args.add_bias                   = bias.has_value() ? 1 : 0;

    CFG* config_map = &cfg_bf16gemm_fp32bf16;

    auto [name, split] = get_heuristic_kernel(
        Mdim, Ndim, Kdim, config_map, arch_id, bpreshuffle, args.add_bias, splitK, kernelName);
    args.splitk              = split;
    AiterAsmKernel* impl_ptr = get_or_load_kernel(name, config_map, SUBM, SUBN);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(A));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    int gdx = (Ndim + SUBN - 1) / SUBN;
    int gdy = (Mdim + SUBM - 1) / SUBM;
    int gdz = split;

    TORCH_CHECK(gdy <= 16, __func__, " gdy (", gdy, ") must be <= 16"); // 16 = 512/32

    // semaphore.fill_(selectedksplit);
    args.ptr_semaphore = (void*)semaphore.data_ptr<uint32_t>();
    // printf("KernelArgs Debug Info:\n");
    // printf("  ptr_D: %p\n", args.ptr_D);
    // printf("  ptr_C: %p\n", args.ptr_C);
    // printf("  ptr_A: %p\n", args.ptr_A);
    // printf("  ptr_B: %p\n", args.ptr_B);
    // printf("  ptr_Bias: %p\n", args.ptr_Bias);
    // printf("  ptr_semaphore: %p\n", args.ptr_semaphore);
    // printf("  alpha: %f\n", args.alpha);
    // printf("  beta: %f\n", args.beta);
    // printf("  stride_D0: %u\n", args.stride_D0);
    // printf("  stride_D1: %u\n", args.stride_D1);
    // printf("  stride_C0: %u\n", args.stride_C0);
    // printf("  stride_C1: %u\n", args.stride_C1);
    // printf("  stride_A0: %u\n", args.stride_A0);
    // printf("  stride_A1: %u\n", args.stride_A1);
    // printf("  stride_B0: %u\n", args.stride_B0);
    // printf("  stride_B1: %u\n", args.stride_B1);
    // printf("  M: %u\n", args.M);
    // printf("  N: %u\n", args.N);
    // printf("  K: %u\n", args.K);
    // printf("  splitk: %u\n", args.splitk);
    // printf("  is_out_b16: %u\n", args.is_out_b16);
    // printf("  add_bias: %u\n", args.add_bias);
    // printf("  Grid: [%d, %d, %d]\n", gdx, gdy, gdz);
    // printf("  Block: [256, 1, 1]\n");
    // printf("  SUBM: %u, SUBN: %u\n", SUBM, SUBN);
    // printf("  Selected Kernel: %s\n", name.c_str());

    size_t arg_size = sizeof(args);
    impl_ptr->launch_kernel({&args, &arg_size, gdx, gdy, gdz, 256, 1, 1, stream});

    return out;
}
