// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <memory>
#include <string>
#include <unordered_map>
#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include "asm_fmoe_2stages_configs.hpp"

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

static CFG *get_cfg(AiterTensor *inp, AiterTensor *out, AiterTensor *w1, QuantType &quant_type, bool do_weight)
{
    int E = w1->size(0);
    int dim1 = w1->size(1);

    if ((inp->dtype() == AITER_DTYPE_fp8) &&
        (w1->dtype() == AITER_DTYPE_fp8) &&
        out->dtype() == AITER_DTYPE_bf16 &&
        quant_type == QuantType::per_Token &&
        do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_doweight_g1u1;
    }
    else if ((inp->dtype() == AITER_DTYPE_fp8) &&
             (w1->dtype() == AITER_DTYPE_fp8) &&
             out->dtype() == AITER_DTYPE_bf16 &&
             quant_type == QuantType::per_Token &&
             !do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_g1u1;
    }
    else if (inp->dtype() == AITER_DTYPE_i8 &&
             w1->dtype() == AITER_DTYPE_i8 &&
             out->dtype() == AITER_DTYPE_bf16 &&
             quant_type == QuantType::per_Token &&
             !do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenInt8_g1u1;
    }
    else if ((inp->dtype() == AITER_DTYPE_fp8) &&
             (w1->dtype() == AITER_DTYPE_fp8) &&
             (out->dtype() == AITER_DTYPE_fp8) &&
             quant_type == QuantType::per_1x128 &&
             !do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1;
    }
    else
    {
        AITER_CHECK(false, __func__ << " Unsupported input_type:" << AiterDtype_to_str(inp->dtype())
                    << ", weight_type:" << AiterDtype_to_str(w1->dtype())
                    << ", out_type:" << AiterDtype_to_str(out->dtype())
                    << ", quant_type:" << static_cast<int>(quant_type) << ", do_weight:" << do_weight);
    }
    return nullptr;
};

std::string get_heuristic_kernel(int m_num, int N, int blockk_size, CFG *cfgs, std::string arch_id)
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
        if (el.first.find(arch_id) != 0)
            continue;
        const auto &cfg = el.second;
        if (cfg.tile_m != blockk_size || N % cfg.tile_n != 0)
        {
            continue;
        }

        tg_num = (N + cfg.tile_n - 1) / cfg.tile_n * m_num;
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

extern "C" __attribute__((visibility("default")))
void moe_stage1_g1u1(
    AiterTensor *input,             // [token_cnt, model_dim] M,K
    AiterTensor *w1,                // [expert, inter_dim*2, model_dim] N,K
    AiterTensor *w2,                // [expert, model_dim, inter_dim]
    AiterTensor *sorted_token_ids,  // [max_num_tokens_padded]
    AiterTensor *sorted_expert_ids, // [max_num_m_blocks]
    AiterTensor *num_valid_ids,     // [1]
    AiterTensor *out,               // [token_cnt, topk, inter_dim*2]
    int inter_dim,
    const char *kernelName,
    int block_m,
    int ksplit,
    int activation,
    int quant_type,
    AiterTensor *a1_scale,      // NULL = not present
    AiterTensor *w1_scale,      // NULL = not present
    AiterTensor *sorted_weights, // NULL = not present
    hipStream_t stream)
{
    int prev_device;
    HIP_CALL(hipGetDevice(&prev_device));
    HIP_CALL(hipSetDevice(input->device_id));

    ActivationType act = static_cast<ActivationType>(activation);
    QuantType qt = static_cast<QuantType>(quant_type);

    CFG *config_map = get_cfg(input, out, w1, qt, sorted_weights != nullptr);
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    int model_dim = input->size(1);
    int hidden_dim = inter_dim;
    int sub_X_cnt = sorted_expert_ids->size(0);
    std::string arch_id = get_gpu_arch();
    std::string knlName = (kernelName != nullptr && kernelName[0] != '\0') ? arch_id + kernelName : "";
    if (knlName.empty())
    {
        knlName = get_heuristic_kernel(sub_X_cnt, inter_dim, block_m, config_map, arch_id);
    }

    AiterAsmKernel *impl_ptr = nullptr;
    auto it = config_map->find(knlName);
    if (it != config_map->end())
    {
        const auto &cfg = it->second;
        const char *name = cfg.knl_name.c_str();
        const char *co_name = cfg.co_name.c_str();

        AITER_CHECK(inter_dim % cfg.tile_n == 0, "ASM kernel " << std::string(name) << " is not supported for inter_dim = " << std::to_string(inter_dim));

        auto result = impl_ptr_map.emplace(name, nullptr);
        if (result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        AITER_CHECK(false, __func__ << " not find kernel " << knlName);

    int token_cnt = input->size(0);
    int topk = out->size(1);

    int dim = w2->size(1);
    int eprt = w1->size(0);
    const auto &cfg = it->second;
    uint32_t sub_GU = cfg.tile_n;
    AITER_CHECK(block_m == cfg.tile_m, __func__ << " kernel: " << cfg.knl_name << " need block_m == " << cfg.tile_m);

    int stride_X = input->stride(0) * input->element_size();
    int stride_GU = dim * w1->element_size();

    int stride_expert_GU = stride_GU * inter_dim;
    int stride_expert_GUDQN = w1_scale ? w1_scale->stride(0) * sizeof(float) : 0;
    int stride_expert_SMTDQN = inter_dim * sizeof(float);
    int stride_O = out->stride(0) * out->element_size();
    if (inter_dim * 2 == w1->size(1))
    {
        stride_expert_GU *= 2;
    }

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_O = out->data_ptr();
    args.ptr_X = input->data_ptr();
    args.ptr_GU = w1->data_ptr();
    args.ptr_XC = num_valid_ids->data_ptr();

    args.ptr_XQ = a1_scale ? a1_scale->data_ptr() : nullptr;
    args.ptr_GUQ = w1_scale ? w1_scale->data_ptr() : nullptr;

    args.ptr_STP = sorted_token_ids->data_ptr();
    args.ptr_SEP = sorted_expert_ids->data_ptr();
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
    args.activation = static_cast<int>(act);
    args.ptr_SW = sorted_weights ? sorted_weights->data_ptr() : nullptr;

    uint32_t k_num = 1 << ksplit;
    AITER_CHECK(model_dim % k_num == 0, __func__ << " Unsupported ksplit for model_dim:" << model_dim << " k_num:" << k_num);

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

    HIP_CALL(hipSetDevice(prev_device));
}
