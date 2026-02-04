// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_fmoe_2stages_configs.hpp"
#include "moe_op.h"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>
struct __attribute__((packed)) KernelArgs
{
    void* ptr_O;
    p2 _p0;
    void* ptr_X;
    p2 _p1;
    void* ptr_GU;
    p2 _p2;
    void* ptr_XC;
    p2 _p3;
    void* ptr_XQ;
    p2 _p4;
    void* ptr_GUQ;
    p2 _p5;
    void* ptr_SMQ;
    p2 _p6;
    void* ptr_STP;
    p2 _p7;
    void* ptr_SEP;
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
    void* ptr_SW;
    p2 _p22;
    unsigned int total_tgs;
    p3 _p23;
    unsigned int ps_deno;
    p3 _p24;
};
struct __attribute__((packed)) Kernel2Args
{
    void* ptr_OBuffer;
    p2 _p0;
    void* ptr_XBuffer;
    p2 _p1;
    void* ptr_DBuffer;
    p2 _p2;
    void* ptr_XCBuffer;
    p2 _p3;
    void* ptr_ScaleXBuffer;
    p2 _p4;
    void* ptr_ScaleDBuffer;
    p2 _p5;
    void* ptr_STPBuffer;
    p2 _p6;
    void* ptr_SEPBuffer;
    p2 _p7;
    unsigned int dim;
    p3 _p8;
    unsigned int hidden_dim;
    p3 _p9;
    unsigned int token_cnt;
    p3 _p10;
    unsigned int eprt_cnt;
    p3 _p11;
    unsigned int stride_X;
    p3 _p12;
    unsigned int stride_D;
    p3 _p13;
    unsigned int stride_O;
    p3 _p14;
    unsigned int stride_expert_D;
    p3 _p15;
    unsigned int stride_expert_scale_D;
    p3 _p16;
    unsigned int topk;
    p3 _p17;
    unsigned int splitk;
    p3 _p18;
    void* ptr_SWBuffer;
    p2 _p19;
};

static CFG* get_cfg(torch::Tensor& inp,
                    torch::Tensor& out,
                    torch::Tensor& w1,
                    QuantType& quant_type,
                    bool do_weight)
{
    int E    = w1.size(0);
    int dim1 = w1.size(1);

    if((inp.scalar_type() == torch_fp8) && (w1.scalar_type() == torch_fp8) &&
       out.scalar_type() == at::ScalarType::BFloat16 && quant_type == QuantType::per_Token &&
       do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_doweight_g1u1;
    }
    else if((inp.scalar_type() == torch_fp8) && (w1.scalar_type() == torch_fp8) &&
            out.scalar_type() == at::ScalarType::BFloat16 && quant_type == QuantType::per_Token &&
            !do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_g1u1;
    }
    else if(inp.scalar_type() == at::ScalarType::Char && w1.scalar_type() == at::ScalarType::Char &&
            out.scalar_type() == at::ScalarType::BFloat16 && quant_type == QuantType::per_Token &&
            !do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenInt8_g1u1;
    }
    else if((inp.scalar_type() == torch_fp8) && (w1.scalar_type() == torch_fp8) &&
            (out.scalar_type() == torch_fp8) && quant_type == QuantType::per_1x128 && !do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1;
    }
    else if(inp.scalar_type() == at::ScalarType::Char && w1.scalar_type() == at::ScalarType::Char &&
            (out.scalar_type() == at::ScalarType::Char) && quant_type == QuantType::per_Token)
    {
        return &cfg_fmoe_stage1_int8_pertokenInt8_g1u1;
    }
    else
    {
        TORCH_CHECK(false,
                    __func__,
                    " Unsupported input_type:",
                    inp.scalar_type(),
                    ", weight_type:",
                    w1.scalar_type(),
                    ", out_type:",
                    out.scalar_type(),
                    ", quant_type:",
                    static_cast<int>(quant_type),
                    ", do_weight:",
                    do_weight);
    }
};

static CFG* get_cfg_stage2(torch::Tensor& inter_states,
                           torch::Tensor& out,
                           torch::Tensor& w2,
                           QuantType& quant_type,
                           bool do_weight)
{
    if(inter_states.scalar_type() == at::ScalarType::Char &&
       w2.scalar_type() == at::ScalarType::Char && out.scalar_type() == at::ScalarType::BFloat16 &&
       quant_type == QuantType::per_Token && do_weight)
    {
        return &cfg_fmoe_stage2_bf16_pertokenInt8_g1u1;
    }
    else
    {
        TORCH_CHECK(false,
                    __func__,
                    " Unsupported input_type:",
                    inter_states.scalar_type(),
                    ", weight_type:",
                    w2.scalar_type(),
                    ", out_type:",
                    out.scalar_type(),
                    ", quant_type:",
                    static_cast<int>(quant_type),
                    ", do_weight:",
                    do_weight);
    }
}

std::string get_heuristic_kernel(int m_num, int N, int blockk_size, CFG* cfgs, std::string arch_id)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu      = dev_prop.multiProcessorCount;
    uint32_t empty_cu    = num_cu;
    uint32_t tg_num      = 0;
    uint32_t round       = 0xffffffff;
    std::string selected = "inter_dim = " + std::to_string(N);

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.tile_m != blockk_size || N % cfg.tile_n != 0)
        {
            continue;
        }

        tg_num               = (N + cfg.tile_n - 1) / cfg.tile_n * m_num;
        uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
        if(local_round < round)
        {
            round    = local_round;
            selected = el.first;
            empty_cu = local_round * num_cu - tg_num;
        }
        else if(local_round == round)
        {
            if(empty_cu > (local_round * num_cu - tg_num))
            {
                round    = local_round;
                selected = el.first;
                empty_cu = local_round * num_cu - tg_num;
            }
        }
    }
    return selected;
}
void moe_stage1_g1u1(
    torch::Tensor& input,             // [token_cnt, model_dim] M,K
    torch::Tensor& w1,                // [expert, inter_dim*2, model_dim] N,K
    torch::Tensor& w2,                // [expert, model_dim, inter_dim]
    torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
    torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
    torch::Tensor& num_valid_ids,     // [1]
    torch::Tensor& out,               // [token_cnt, topk, inter_dim*2]
    int inter_dim,
    std::string& kernelName,
    int block_m,
    int ksplit                            = 0,
    ActivationType activation             = ActivationType::Silu,
    QuantType quant_type                  = QuantType::No,
    std::optional<torch::Tensor> a1_scale = std::nullopt, // [token_cnt, 1], token scale
    std::optional<torch::Tensor> w1_scale = std::nullopt, // [expert, 1, inter_dim], gate(up) scale
    std::optional<torch::Tensor> fc2_smooth_scale = std::nullopt,
    std::optional<torch::Tensor> fc2_scale        = std::nullopt,
    std::optional<torch::Tensor> sorted_weights =
        std::nullopt // [max_num_tokens_padded], do_weight==true need
)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    CFG* config_map = get_cfg(input, out, w1, quant_type, sorted_weights.has_value());
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    int model_dim  = input.size(-1);
    int hidden_dim = inter_dim;

    // Handle [TOPK, BATCH, DIM] vs [BATCH, TOPK, DIM]
    // token_cnt should be BATCH count based on Host Code implication
    int token_cnt = input.size(-2);
    int topk = out.size(1);
    int eprt = w1.size(0);

    // User specified logic for gdy / sub_X_cnt calculation
    // sub_X -> block_m
    // batch -> token_cnt
    // sz_sep -> gdy
    long long sz_stp = (long long)topk * token_cnt + (long long)eprt * block_m - topk;
    sz_stp           = (sz_stp + block_m - 1) / block_m;
    int sub_X_cnt    = sorted_expert_ids.size(0);

    std::string arch_id = get_gpu_arch();
    kernelName          = !kernelName.empty() ? arch_id + kernelName : "";
    if(kernelName.empty())
    {
        kernelName = get_heuristic_kernel(sub_X_cnt, inter_dim, block_m, config_map, arch_id);
    }

    AiterAsmKernel* impl_ptr = nullptr;
    auto it                  = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();

        TORCH_CHECK(inter_dim % cfg.tile_n == 0,
                    "ASM kernel " + std::string(name) +
                        " is not supported for inter_dim = " + std::to_string(inter_dim));

        auto result = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel " + kernelName);

    // const char *enable_vskip = std::getenv("AITER_ENABLE_VSKIP");
    const auto& cfg = it->second;
    uint32_t sub_GU = cfg.tile_n;
    TORCH_CHECK(block_m == cfg.tile_m,
                __func__,
                " kernel: ",
                cfg.knl_name,
                " need block_m == ",
                cfg.tile_m);

    int stride_X = input.stride(-2);
    // if(input.dim() == 3 && input.size(0) == out.size(1) && input.size(1) == out.size(0))
    // {
    //     stride_X = input.stride(1) * input.element_size();
    // }
    // else
    // {
    //     stride_X = input.stride(0) * input.element_size();
    // }
    int stride_GU = model_dim * w1.element_size();

    int stride_expert_GU    = stride_GU * inter_dim;
    int stride_expert_GUDQN = w1_scale.has_value() ? w1_scale.value().stride(0) * sizeof(float) : 0;
    int stride_expert_SMTDQN = inter_dim * sizeof(float);
    int stride_O             = topk * inter_dim * out.element_size();

    if(inter_dim * 2 == w1.size(1))
    {
        stride_expert_GU *= 2;
    }

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_O      = out.data_ptr();
    args.ptr_X      = input.data_ptr();
    args.ptr_GU     = w1.data_ptr();
    args.ptr_XC     = num_valid_ids.data_ptr();

    args.ptr_XQ  = a1_scale.has_value() ? a1_scale.value().data_ptr() : nullptr;
    args.ptr_GUQ = w1_scale.has_value() ? w1_scale.value().data_ptr() : nullptr;
    args.ptr_SMQ = fc2_smooth_scale.has_value() ? fc2_smooth_scale.value().data_ptr() : nullptr;

    args.ptr_STP    = sorted_token_ids.data_ptr();
    args.ptr_SEP    = sorted_expert_ids.data_ptr();
    args.dim        = model_dim;
    args.hidden_dim = inter_dim;
    args.token_cnt  = token_cnt;
    args.eprt_cnt   = eprt;
    args.Xs         = stride_X;
    args.GUs        = stride_GU;
    args.Os         = stride_O;
    args.eGUs       = stride_expert_GU;
    args.eGUQs      = stride_expert_GUDQN;
    args.eSMQs      = stride_expert_SMTDQN;
    args.topk       = topk;
    args.splitk     = ksplit;
    args.activation = static_cast<int>(activation);
    args.ptr_SW     = sorted_weights.has_value() ? sorted_weights.value().data_ptr() : nullptr;

    uint32_t k_num = 1 << ksplit;
    TORCH_CHECK(model_dim % k_num == 0,
                __func__,
                " Unsupported ksplit for model_dim:",
                model_dim,
                " k_num:",
                k_num);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size,
                      HIP_LAUNCH_PARAM_END};

    int bdx = 256;
    int gdx = ((hidden_dim + sub_GU - 1) / sub_GU);
    int gdy = sz_stp; // sub_X_cnt;
    int gdz = k_num;

    printf("#### stage1 arg start ############\n");
    std::cout << "dim:" << args.dim << std::endl;
    std::cout << "hidden:" << args.hidden_dim << std::endl;
    std::cout << "token:" << args.token_cnt << std::endl;
    std::cout << "eprt:" << args.eprt_cnt << std::endl;
    std::cout << "Xs:" << args.Xs << std::endl;
    std::cout << "GUs:" << args.GUs << std::endl;
    std::cout << "Os:" << args.Os << std::endl;
    std::cout << "eGUs:" << args.eGUs << std::endl;
    std::cout << "GUQs:" << args.eGUQs << std::endl;
    std::cout << "SMQs:" << args.eSMQs << std::endl;
    std::cout << "topk:" << args.topk << std::endl;
    std::cout << "splitk:" << args.splitk << std::endl;
    printf("gdx:%d, gdy:%d, gdz:%d, tgs:%d\n", gdx, gdy, gdz, sub_X_cnt * gdx * gdz);

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

void moe_stage2_g1u1(
    torch::Tensor& inter_states,      // [token_cnt, topk, inter_dim]
    torch::Tensor& w1,                // [expert, inter_dim*2, model_dim] N,K
    torch::Tensor& w2,                // [expert, model_dim, inter_dim]
    torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
    torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
    torch::Tensor& num_valid_ids,     // [1]
    torch::Tensor& out,               // [token_cnt, topk, model_dim]
    int topk,
    std::string& kernelName,
    int block_m,
    std::optional<torch::Tensor> w2_scale = std::nullopt, // [expert, 1, dim], down scale
    std::optional<torch::Tensor> a2_scale = std::nullopt, // [token_cnt, 1], inter scale
    std::optional<torch::Tensor> sorted_weights =
        std::nullopt, // [max_num_tokens_padded], do_weight==true need
    QuantType quant_type      = QuantType::No,
    ActivationType activation = ActivationType::Silu,
    int splitk                = 0)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inter_states));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    CFG* config_map = get_cfg_stage2(inter_states, out, w2, quant_type, sorted_weights.has_value());
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;

    int inter_dim = inter_states.size(-1); // inter_states: [..., inter_dim]

    int sub_X_cnt;
    if(num_valid_ids.numel() > 0)
    {
        int valid_token_cnt = num_valid_ids[0].item<int>();
        sub_X_cnt           = (valid_token_cnt + block_m - 1) / block_m;
    }
    else
    {
        sub_X_cnt = sorted_expert_ids.size(0);
    }
    std::string arch_id = get_gpu_arch();
    kernelName          = !kernelName.empty() ? arch_id + kernelName : "";
    if(kernelName.empty())
    {
        kernelName = get_heuristic_kernel(sub_X_cnt, out.size(-1), block_m, config_map, arch_id);
    }

    AiterAsmKernel* impl_ptr = nullptr;
    auto it                  = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();

        auto result = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel " + kernelName);

    int token_cnt;

    if(inter_states.dim() == 3)
    {
        token_cnt = inter_states.size(0);
    }
    else
    {
        token_cnt = inter_states.size(0);
    }

    int dim         = out.size(-1);
    int eprt        = w2.size(0);
    const auto& cfg = it->second;
    uint32_t sub_D  = cfg.tile_n;
    TORCH_CHECK(block_m == cfg.tile_m,
                __func__,
                " kernel: ",
                cfg.knl_name,
                " need block_m == ",
                cfg.tile_m);

    int stride_X = inter_dim * inter_states.element_size();

    int stride_D = inter_dim * w2.element_size();

    int stride_Scale_X = a2_scale.has_value() ? a2_scale.value().stride(0) * sizeof(float) : 0;

    int stride_expert_D = inter_dim * dim * w2.element_size();

    int stride_expert_scale_D = w2_scale.has_value() ? dim * sizeof(float) : 0;

    int dbl_o    = (splitk > 1) ? 2 : 1;
    int stride_O = dim * out.element_size() * dbl_o;

    Kernel2Args args;
    size_t arg_size = sizeof(args);
    memset(&args, 0, sizeof(args));
    args.ptr_OBuffer  = out.data_ptr();
    args.ptr_XBuffer  = inter_states.data_ptr();
    args.ptr_DBuffer  = w2.data_ptr();
    args.ptr_XCBuffer = num_valid_ids.data_ptr();

    args.ptr_ScaleXBuffer = a2_scale.has_value() ? a2_scale.value().data_ptr() : nullptr;
    args.ptr_ScaleDBuffer = w2_scale.has_value() ? w2_scale.value().data_ptr() : nullptr;

    args.ptr_STPBuffer         = sorted_token_ids.data_ptr();
    args.ptr_SEPBuffer         = sorted_expert_ids.data_ptr();
    args.dim                   = dim;       // Output dim (model dim)
    args.hidden_dim            = inter_dim; // Input dim (inter dim)
    args.token_cnt             = token_cnt;
    args.eprt_cnt              = eprt;
    args.stride_X              = stride_X;
    args.stride_D              = stride_D;
    args.stride_O              = stride_O;
    args.stride_expert_D       = stride_expert_D;
    args.stride_expert_scale_D = stride_expert_scale_D;
    args.topk                  = topk;
    args.splitk                = splitk;
    args.ptr_SWBuffer = sorted_weights.has_value() ? sorted_weights.value().data_ptr() : nullptr;

    uint32_t k_num = 1;

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size,
                      HIP_LAUNCH_PARAM_END};

    int bdx = 256;
    int gdx = ((dim + sub_D - 1) / sub_D);
    int gdy = sub_X_cnt;
    int gdz = k_num;

    printf("#### stage2 arg start ############\n ");
    printf("args.ptr_OBuffer  = %p\n", args.ptr_OBuffer);
    printf("args.ptr_XBuffer  = %p\n", args.ptr_XBuffer);
    printf("args.ptr_DBuffer  = %p\n", args.ptr_DBuffer);
    printf("args.ptr_XCBuffer = %p\n", args.ptr_XCBuffer);
    printf("args.ptr_ScaleXBuffer = %p\n", args.ptr_ScaleXBuffer);
    printf("args.ptr_ScaleDBuffer = %p\n", args.ptr_ScaleDBuffer);
    printf("args.ptr_STPBuffer = %p\n", args.ptr_STPBuffer);
    printf("args.ptr_SEPBuffer = %p\n", args.ptr_SEPBuffer);
    printf("args.dim = %u\n", args.dim);
    printf("args.hidden_dim = %u\n", args.hidden_dim);
    printf("args.token_cnt = %u\n", args.token_cnt);
    printf("args.eprt_cnt = %u\n", args.eprt_cnt);
    printf("args.stride_X = %u\n", args.stride_X);
    printf("args.stride_D = %u\n", args.stride_D);
    printf("args.stride_O = %u\n", args.stride_O);
    printf("args.stride_expert_D = %u\n", args.stride_expert_D);
    printf("args.stride_expert_scale_D = %u\n", args.stride_expert_scale_D);
    printf("args.topk = %u\n", args.topk);
    printf("args.splitk = %u\n", args.splitk);
    printf("args.ptr_SWBuffer = %p\n", args.ptr_SWBuffer);
    printf("gdx = %d\n", gdx);
    printf("gdy = %d\n", gdy);
    printf("gdz = %d\n", gdz);

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