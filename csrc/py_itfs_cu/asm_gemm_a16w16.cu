// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_bf16gemm_configs.hpp"
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <hip/hip_runtime.h>
#include <torch/all.h>

// start to prepare the input and output buffer
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
    p2 _p17;
};

std::tuple<std::string, int>
get_heuristic_kernel(int M,
                     int N,
                     int K,
                     CFG* cfgs,
                     std::optional<int> splitk             = std::nullopt,
                     std::optional<std::string> kernelName = std::nullopt)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu = dev_prop.multiProcessorCount;
    // printf("num_cu: %d\n", num_cu);
    uint32_t empty_cu      = num_cu;
    uint32_t pure_tg_num   = 0;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0;
    int oob                = M;

    std::string selectedKernelName = "";
    int selectedsplitK             = 1;

    for(const auto& el : *cfgs)
    {
        const auto& cfg = el.second;
        if(kernelName.has_value() && kernelName.value() != el.first)
            continue;
        if(N % cfg.tileN == 0)
        {
            // 1. select splitK
            int split_K = 1;
            if(splitk.has_value())
                split_K = splitk.value();
            else // auto select
            {
                pure_tg_num =
                    ((M + cfg.tileM - 1) / cfg.tileM) * (N / cfg.tileN); // M-orient support OOB
                if(pure_tg_num < num_cu)
                {
                    int max_split = num_cu / pure_tg_num;
                    for(int i = max_split; i >= 1; i--)
                    {
                        if(K % (i * 64) == 0)
                        {
                            split_K = i;
                            break;
                        }
                    }
                }
            }

            uint32_t tg_num = pure_tg_num * split_K;
            // 2. better or not
            uint32_t local_round         = (tg_num + num_cu - 1) / num_cu;
            float local_compute2mem_effi = cfg.tileM * cfg.tileN / (cfg.tileM + cfg.tileN);
            bool is_earlier_round        = (local_round < round);
            bool is_same_round           = (local_round == round);
            bool has_sufficient_empty_cu = (empty_cu > (local_round * num_cu - tg_num));
            bool has_same_empty_cu       = empty_cu == (local_round * num_cu - tg_num);
            bool has_better_efficiency   = (local_compute2mem_effi > compute2mem_effi);
            // printf("oob %d,  tielM: %d\n", oob, cfg.tileM);
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

torch::Tensor gemm_a16w16_asm(torch::Tensor& A,   // A:[M, K] bf16
                              torch::Tensor& B,   // B:[N, K] bf16
                              torch::Tensor& out, // Out:[M, N] f32
                              std::optional<torch::Tensor> bias,
                              std::optional<int> splitK,
                              std::optional<std::string> kernelName)
{
    TORCH_CHECK(out.dtype() == torch::ScalarType::Float,
                "GEMM A16W16 asm only support Float32 output now!");

    // 1. prepare args
    int Mdim = A.size(0);
    int Ndim = B.size(0);
    int Kdim = A.size(1);

    unsigned int SUBM = 64;
    unsigned int SUBN = 64;
    float alpha       = 1.0;
    float beta        = 0.0;
    int szA           = Mdim * Kdim;
    int szB           = Kdim * Ndim;
    int szC           = Mdim * Ndim;
    int sz_A_pad      = 0;
    int sz_B_pad      = 0;
    int sz_C_pad      = 0;
    int strideD0      = 0;
    int strideD1      = 0;
    int strideC0      = 0;
    int strideC1      = 0;
    int strideA0      = 0;
    int strideA1      = 0;
    int strideB0      = 0;
    int strideB1      = 0;
    // A row major, B col major, C row major
    strideA0 = strideA1 = Kdim * 2; // in bytes
    strideB0 = strideB1 = Kdim * 2;
    strideC0 = strideC1 = strideD0 = strideD1 = Ndim * 4; // inbytes

    szA += sz_A_pad;
    szB += sz_B_pad;
    szC += sz_C_pad;

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_D      = (void*)out.data_ptr();
    // args.ptr_C        = bias.has_value() ? (void*)bias.value().data_ptr() : nullptr;
    args.ptr_C     = (void*)NULL;
    args.ptr_A     = (void*)A.data_ptr();
    args.ptr_B     = (void*)B.data_ptr();
    args.alpha     = alpha;
    args.beta      = beta;
    args.stride_C0 = strideC0;
    args.stride_A0 = strideA0;
    args.stride_B0 = strideB0;
    args.M         = Mdim;
    args.N         = Ndim;
    args.K         = Kdim;

    // args.stride_D0 = 25;
    // args.stride_D1 = 80;
    // args.stride_C1 = 3;
    // args.stride_A1 = 124;

    // 2. select kl
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    AiterAsmKernel* impl_ptr = nullptr;
    CFG* config_map          = &cfg_bf16gemm_outf32;

    // 2.1 static dict
    std::string selectedKernelName;
    int selectedksplit;
    auto it_sel        = get_heuristic_kernel(Mdim,
                                       Ndim,
                                       Kdim,
                                       config_map,
                                       splitK.has_value() ? splitK : std::nullopt,
                                       kernelName.has_value() ? kernelName : std::nullopt);
    selectedKernelName = std::get<0>(it_sel);
    selectedksplit     = std::get<1>(it_sel);

    args.splitk = selectedksplit;
    // printf("=== KernelArgs Important Parameters ===\n");
    // printf("ptr_D: %p\n", args.ptr_D);
    // printf("ptr_A: %p\n", args.ptr_A);
    // printf("ptr_B: %p\n", args.ptr_B);
    // printf("alpha: %f\n", args.alpha);
    // printf("beta: %f\n", args.beta);
    // printf("stride_D0: %u\n", args.stride_D0);
    // printf("stride_D1: %u\n", args.stride_D1);
    // printf("stride_C0: %u\n", args.stride_C0);
    // printf("stride_C1: %u\n", args.stride_C1);
    // printf("stride_A0: %u\n", args.stride_A0);
    // printf("stride_A1: %u\n", args.stride_A1);
    // printf("stride_B0: %u\n", args.stride_B0);
    // printf("stride_B1: %u\n", args.stride_B1);
    // printf("M: %u\n", args.M);
    // printf("N: %u\n", args.N);
    // printf("K: %u\n", args.K);
    // printf("splitk: %u\n", args.splitk);
    // printf("=======================================\n");

    auto it_kl = config_map->find(selectedKernelName);
    if(it_kl != config_map->end())
    {
        const auto& cfg     = it_kl->second;
        const char* name    = cfg.name.c_str();
        const char* co_name = cfg.co_name.c_str();
        SUBM                = cfg.tileM;
        SUBN                = cfg.tileN;
        auto result         = impl_ptr_map.emplace(name, nullptr); // insert new kl.
        if(result.second)                                          // emplace successfully
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        impl_ptr = result.first->second.get();
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel~ " + selectedKernelName);

    // 3. launch kl
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int bdx = 256;
    int gdx = (Ndim + SUBN - 1) / SUBN;
    int gdy = ((Mdim + SUBM - 1) / SUBM);
    int gdz = 1;

    if(selectedksplit > 1)
    {
        out.zero_();
        int k_per_tg = Kdim / selectedksplit;
        gdz          = selectedksplit;
    }

    // printf("argsize: %zu\n", arg_size);
    // printf("gdx: %d\n", gdx);
    // printf("gdy: %d\n", gdy);
    // printf("gdz: %d\n", gdz);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx, // gdx
                             gdy, // gdy
                             gdz, // gdz
                             256, // bdx: 4 wv64
                             1,   // bdy
                             1,   // bdz
                             stream});

    // 4. return out
    return out;
}
