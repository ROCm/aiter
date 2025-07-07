// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "moe_op.h"
#include "asm_moe_2stage_configs.hpp"
#include "py_itfs_common.h"

struct __attribute__((packed)) KernelArgs
{
    void *ptr_O;
    p2 _p0;
    void *ptr_X;
    p2 _p1;
    void *ptr_GU;
    p2 _p2;
    void *ptr_XC;
    p2 _p3;
    void *ptr_XQ;
    p2 _p4;
    void *ptr_GUQ;
    p2 _p5;
    void *ptr_SMQ;
    p2 _p6;
    void *ptr_STP;
    p2 _p7;
    void *ptr_SEP;
    p2 _p8;
    unsigned int dim;
    p3 _p9;
    unsigned int hidden_dim;
    p3 _p10;
    unsigned int token_cnt;
    p3 _p11;
    unsigned int eprt_cnt;
    p3 _p12;
    unsigned int Xs;
    p3 _p13;
    unsigned int GUs;
    p3 _p14;
    unsigned int Os;
    p3 _p15;
    unsigned int eGUs;
    p3 _p16;
    unsigned int eGUQs;
    p3 _p17;
    unsigned int eSMQs;
    p3 _p18;
    unsigned int topk;
    p3 _p19;
    unsigned int splitk;
    p3 _p20;
    unsigned int activation;
    p3 _p21;
    void *ptr_SW;
    p2 _p22;
};

static CFG *get_cfg(torch::Tensor &inp, torch::Tensor &out, torch::Tensor &w1, QuantType &quant_type, bool do_weight)
{
    int E = w1.size(0);
    int dim1 = w1.size(1);

    if ((inp.scalar_type() == torch_fp8) &&
        (w1.scalar_type() == torch_fp8) &&
        out.scalar_type() == at::ScalarType::BFloat16 &&
        quant_type == QuantType::per_Token &&
        do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_doweight_g1u1;
    }
    else if ((inp.scalar_type() == torch_fp8) &&
             (w1.scalar_type() == torch_fp8) &&
             out.scalar_type() == at::ScalarType::BFloat16 &&
             quant_type == QuantType::per_Token)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_g1u1;
    }
    else if (inp.scalar_type() == at::ScalarType::Char &&
             w1.scalar_type() == at::ScalarType::Char &&
             out.scalar_type() == at::ScalarType::BFloat16 &&
             quant_type == QuantType::per_Token)
    {
        return &cfg_fmoe_stage1_bf16_pertokenInt8_g1u1;
    }
    else if ((inp.scalar_type() == torch_fp8) &&
             (w1.scalar_type() == torch_fp8) &&
             (out.scalar_type() == torch_fp8) &&
             quant_type == QuantType::per_128x128)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1;
    }
    else
    {
        TORCH_CHECK(false, __func__, " Unsupported input_type:", inp.scalar_type(), ", weight_type:", w1.scalar_type(),
                    ", out_type:", out.scalar_type(), ", quant_type:", static_cast<int>(quant_type), ", do_weight:", do_weight);
    }
};

std::string get_heuristic_kernel(int m_num, int N, int blockk_size, CFG *cfgs)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu = dev_prop.multiProcessorCount;
    uint32_t empty_cu = num_cu;
    uint32_t tg_num = 0;
    uint32_t round = 0xffffffff;
    std::string selected = "inter_dim = " + std::to_string(N);

    for (const auto &el : *cfgs)
    {
        const auto &cfg = el.second;
        if (cfg.tile_M != blockk_size || N % cfg.tile_N != 0)
        {
            continue;
        }

        tg_num = (N + cfg.tile_N - 1) / cfg.tile_N * m_num;
        uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
        if (local_round < round)
        {
            round = local_round;
            selected = el.first;
            empty_cu = local_round * num_cu - tg_num;
        }
        else if (local_round == round)
        {
            if (empty_cu > (local_round * num_cu - tg_num))
            {
                round = local_round;
                selected = el.first;
                empty_cu = local_round * num_cu - tg_num;
            }
        }
    }
    return selected;
}
void moe_stage1_g1u1(
    torch::Tensor &input,             // [token_cnt, model_dim] M,K
    torch::Tensor &w1,                // [expert, inter_dim*2, model_dim] N,K
    torch::Tensor &w2,                // [expert, model_dim, inter_dim]
    torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
    torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
    torch::Tensor &num_valid_ids,     // [1]
    torch::Tensor &out,               // [token_cnt, topk, inter_dim*2]
    int inter_dim,
    std::string &kernelName,
    int block_m,
    int ksplit = 0,
    ActivationType activation = ActivationType::Silu,
    QuantType quant_type = QuantType::No,
    std::optional<torch::Tensor> a1_scale = std::nullopt,      // [token_cnt, 1], token scale
    std::optional<torch::Tensor> w1_scale = std::nullopt,      // [expert, 1, inter_dim], gate(up) scale
    std::optional<torch::Tensor> sorted_weights = std::nullopt // [max_num_tokens_padded], do_weight==true need
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    CFG *config_map = get_cfg(input, out, w1, quant_type, sorted_weights.has_value());
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    int model_dim = input.size(1);
    int hidden_dim = inter_dim;
    int sub_X_cnt = sorted_expert_ids.size(0);
    if (kernelName.empty())
    {
        kernelName = get_heuristic_kernel(sub_X_cnt, inter_dim, block_m, config_map);
    }

    AiterAsmKernel *impl_ptr = nullptr;
    auto it = config_map->find(kernelName);
    if (it != config_map->end())
    {
        const auto &cfg = it->second;
        const char *name = cfg.name.c_str();
        const char *co_name = cfg.co_name.c_str();

        TORCH_CHECK(inter_dim % cfg.tile_N == 0, "ASM kernel " + std::string(name) + " is not supported for inter_dim = " + std::to_string(inter_dim));

        auto result = impl_ptr_map.emplace(name, nullptr);
        if (result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel " + kernelName);

    int token_cnt = input.size(0);
    int topk = out.size(1);

    // const char *enable_vskip = std::getenv("AITER_ENABLE_VSKIP");

    int dim = w2.size(1);
    int eprt = w1.size(0);
    const auto &cfg = it->second;
    uint32_t sub_GU = cfg.tile_N;
    TORCH_CHECK(block_m == cfg.tile_M, __func__, " kernel: ", cfg.name, " need block_m == ", cfg.tile_M);

    int stride_X = input.stride(0) * input.element_size();
    int stride_GU = dim * w1.element_size();

    int stride_expert_GU = stride_GU * inter_dim;
    int stride_expert_GUDQN = w1_scale.has_value() ? w1_scale.value().stride(0) * sizeof(float) : 0;
    int stride_expert_SMTDQN = inter_dim * sizeof(float);
    int stride_O = out.stride(0) * out.element_size();
    if (inter_dim * 2 == w1.size(1))
    {
        stride_expert_GU *= 2;
    }

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_O = out.data_ptr();
    args.ptr_X = input.data_ptr();
    args.ptr_GU = w1.data_ptr();
    args.ptr_XC = num_valid_ids.data_ptr();

    args.ptr_XQ = a1_scale.has_value() ? a1_scale.value().data_ptr() : nullptr;
    args.ptr_GUQ = w1_scale.has_value() ? w1_scale.value().data_ptr() : nullptr;
    // args.ptr_SMQ = w2_smooth_qnt.has_value() ? w2_smooth_qnt.value().data_ptr() : nullptr;

    args.ptr_STP = sorted_token_ids.data_ptr();
    args.ptr_SEP = sorted_expert_ids.data_ptr();
    args.dim = dim;
    args.hidden_dim = inter_dim;
    args.token_cnt = token_cnt;
    args.eprt_cnt = eprt;
    args.Xs = stride_X;
    args.GUs = stride_GU;
    args.Os = stride_O;
    args.eGUs = stride_expert_GU;
    args.eGUQs = stride_expert_GUDQN;
    args.eSMQs = stride_expert_SMTDQN;
    args.topk = topk;
    args.splitk = ksplit;
    args.activation = static_cast<int>(activation);
    args.ptr_SW = sorted_weights.has_value() ? sorted_weights.value().data_ptr() : nullptr;

    uint32_t k_num = 1 << ksplit;
    TORCH_CHECK(model_dim % k_num == 0, __func__, " Unsupported ksplit for model_dim:", model_dim, " k_num:", k_num);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};

    int bdx = 256;
    int gdx = ((hidden_dim + sub_GU - 1) / sub_GU);
    int gdy = sub_X_cnt;
    int gdz = k_num;

    // std::cout << "dim:" << args.dim << std::endl;
    // std::cout << "hidden:" << args.hidden_dim << std::endl;
    // std::cout << "token:" << args.token_cnt << std::endl;
    // std::cout << "eprt:" << args.eprt_cnt << std::endl;
    // std::cout << "Xs:" << args.Xs << std::endl;
    // std::cout << "GUs:" << args.GUs << std::endl;
    // std::cout << "Os:" << args.Os << std::endl;
    // std::cout << "GUs:" << args.eGUs << std::endl;
    // std::cout << "GUQs:" << args.eGUQs << std::endl;
    // std::cout << "SMQs:" << args.eSMQs << std::endl;
    // std::cout << "topk:" << args.topk << std::endl;
    // std::cout << "splitk:" << args.splitk << std::endl;
    // printf("gdx:%d, gdy:%d, gdz:%d, tgs:%d\n", gdx, gdy, gdz, sub_X_cnt * gdx * gdz);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx, // gdx
                             gdy, // gdy
                             gdz, // gdz
                             bdx, // bdx: 4 wv64
                             1,   // bdy
                             1,   // bdz
                             stream});
}
