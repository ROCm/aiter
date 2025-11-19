// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_moe_2stage_configs.hpp"
#include "moe_op.h"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
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
};

struct __attribute__((packed)) KernelStage2Args
{
    void* o_buf;
    p2 _p0;
    void* a_buf;
    p2 _p1;
    void* tk_num_buf;
    p2 _p2;
    void* b_buf;
    p2 _p3;
    void* as_buf;
    p2 _p4;
    void* bs_buf;
    p2 _p5;
    void* tk_buf;
    p2 _p6;
    void* w_buf;
    p2 _p7;
    void* expt_buf;
    p2 _p8;
    int inter_dim;
    p3 _p9;
    int model_dim;
    p3 _p10;
    int tokens;
    p3 _p11;
    int experts;
    p3 _p12;
    int stride_a;
    p3 _p13;
    int stride_b;
    p3 _p14;
    int stride_o;
    p3 _p15;
    int stride_expt_b;
    p3 _p16;
    int stride_expt_bs;
    p3 _p17;
    int topks;
    p3 _p18;
    int split_k;
    p3 _p19;
    void* o_flat_buf;
    p2 _p20;
    void* dbg_i32;
    p2 _p21;
    void* dbg_f32;
    p2 _p22;
    void* dbg_f16;
    p2 _p23;
    void* dbg_lds;
    p2 _p24;
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
    else if((inp.scalar_type() == torch::kFloat4_e2m1fn_x2) ||
            (inp.scalar_type() == torch::kUInt8) &&
                (out.scalar_type() == at::ScalarType::BFloat16) &&
                quant_type == QuantType::per_1x32)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp4_blockscale_g1u1;
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

static CFG* get_stage2_cfg(torch::Tensor& inp,
                           torch::Tensor& out,
                           torch::Tensor& w2,
                           QuantType& quant_type,
                           bool do_weight)
{
    int E = w2.size(0);

    if((inp.scalar_type() == torch::kFloat4_e2m1fn_x2) ||
       (inp.scalar_type() == torch::kUInt8) && (out.scalar_type() == at::ScalarType::BFloat16) &&
           quant_type == QuantType::per_1x32)
    {
        return &cfg_fmoe_stage2_bf16_pertokenFp4_blockscale_g1u1;
    }
    else
    {
        TORCH_CHECK(false,
                    __func__,
                    " Unsupported input_type:",
                    inp.scalar_type(),
                    ", weight_type:",
                    w2.scalar_type(),
                    ", out_type:",
                    out.scalar_type(),
                    ", quant_type:",
                    static_cast<int>(quant_type),
                    ", do_weight:",
                    do_weight);
    }
};

std::string get_heuristic_kernel(int m_num, int N, int blockk_size, CFG* cfgs)
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
        const auto& cfg = el.second;
        if(cfg.tile_M != blockk_size || N % cfg.tile_N != 0)
        {
            continue;
        }

        tg_num               = (N + cfg.tile_N - 1) / cfg.tile_N * m_num;
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
    std::optional<torch::Tensor> sorted_weights =
        std::nullopt // [max_num_tokens_padded], do_weight==true need
)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    CFG* config_map = get_cfg(input, out, w1, quant_type, sorted_weights.has_value());
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    int model_dim  = input.size(1);
    int hidden_dim = inter_dim;
    int sub_X_cnt  = sorted_expert_ids.size(0);
    if(kernelName.empty())
    {
        kernelName = get_heuristic_kernel(sub_X_cnt, inter_dim, block_m, config_map);
    }

    AiterAsmKernel* impl_ptr = nullptr;
    auto it                  = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.name.c_str();
        const char* co_name = cfg.co_name.c_str();

        TORCH_CHECK(inter_dim % cfg.tile_N == 0,
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

    int token_cnt = input.size(0);
    int topk      = out.size(1);

    // const char *enable_vskip = std::getenv("AITER_ENABLE_VSKIP");

    int dim         = w2.size(1);
    int eprt        = w1.size(0);
    const auto& cfg = it->second;
    uint32_t sub_GU = cfg.tile_N;
    TORCH_CHECK(
        block_m == cfg.tile_M, __func__, " kernel: ", cfg.name, " need block_m == ", cfg.tile_M);

    int stride_X  = input.stride(0) * input.element_size();
    int stride_GU = dim * w1.element_size();

    int stride_expert_GU =
        input.scalar_type() == torch::kFloat4_e2m1fn_x2 || input.scalar_type() == torch::kUInt8
            ? hidden_dim * dim / 2
            : stride_GU * inter_dim;
    int stride_expert_GUDQN =
        input.scalar_type() == torch::kFloat4_e2m1fn_x2 || input.scalar_type() == torch::kUInt8
            ? hidden_dim * 2 * (dim / 32)
        : w1_scale.has_value() ? w1_scale.value().stride(0) * sizeof(float)
                               : 0;
    int stride_expert_SMTDQN = inter_dim * sizeof(float);
    int stride_O             = out.stride(0) * out.element_size();
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
    // args.ptr_SMQ = w2_smooth_qnt.has_value() ? w2_smooth_qnt.value().data_ptr() : nullptr;

    args.ptr_STP    = sorted_token_ids.data_ptr();
    args.ptr_SEP    = sorted_expert_ids.data_ptr();
    args.dim        = dim;
    args.hidden_dim = inter_dim;
    args.token_cnt  = token_cnt;
    args.eprt_cnt   = eprt;
    args.Xs         = stride_X;
    args.GUs =
        input.scalar_type() == torch::kFloat4_e2m1fn_x2 || input.scalar_type() == torch::kUInt8
            ? dim / 2
            : stride_GU;
    args.Os    = stride_O;
    args.eGUs  = stride_expert_GU;
    args.eGUQs = stride_expert_GUDQN;
    args.eSMQs =
        input.scalar_type() == torch::kFloat4_e2m1fn_x2 || input.scalar_type() == torch::kUInt8
            ? 0
            : stride_expert_SMTDQN;
    args.topk       = topk;
    args.splitk     = ksplit;
    args.activation = static_cast<int>(activation);
    args.ptr_SW =
        input.scalar_type() == torch::kFloat4_e2m1fn_x2 || input.scalar_type() == torch::kUInt8
            ? nullptr
        : sorted_weights.has_value() ? sorted_weights.value().data_ptr()
                                     : nullptr;

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

void moe_stage2_g1u1(
    torch::Tensor& input,             // [token, topK, inter_dim] M,toK K
    torch::Tensor& w1,                // [expert, inter_dim*2, model_dim/2] E,K,N/2
    torch::Tensor& w2,                // [expert, model_dim, inter_dim/2] E,N,K/2
    torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
    torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
    torch::Tensor& num_valid_ids,     // [1]
    torch::Tensor& out,               // [token_cnt, inter_dim]
    int inter_dim,
    std::string& kernelName,
    int block_m,
    int ksplit                            = 0,
    ActivationType activation             = ActivationType::Silu,
    QuantType quant_type                  = QuantType::No,
    std::optional<torch::Tensor> a2_scale = std::nullopt, // [token_cnt, 1], token scale
    std::optional<torch::Tensor> w2_scale = std::nullopt, // [expert, 1, model_dim], down scale
    std::optional<torch::Tensor> sorted_weights =
        std::nullopt // [max_num_tokens_padded], do_weight==true need
)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    CFG* config_map = get_stage2_cfg(input, out, w2, quant_type, sorted_weights.has_value());
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    int N          = w2.size(1);
    int K          = inter_dim;
    int model_dim  = N;
    int hidden_dim = inter_dim;
    int sub_X_cnt  = sorted_expert_ids.size(0);
    int token_cnt  = input.size(0);
    int topk       = input.size(1);
    int eprt       = w1.size(0);
    // prepare kernel args
    int stride_o             = model_dim * out.element_size();    // ROW
    int stride_a             = hidden_dim * input.element_size(); // ROW
    int stride_b             = hidden_dim * w2.element_size();    // COL
    int stride_exprt_b       = K * N;                             // COL
    int stride_exprt_b_scale = N * (K / 32);                      // COL

    stride_a /= 2;
    stride_b /= 2;
    stride_exprt_b /= 2;

    if(kernelName.empty())
    {
        kernelName = get_heuristic_kernel(sub_X_cnt, hidden_dim, block_m, config_map);
    }

    AiterAsmKernel* impl_ptr = nullptr;
    auto it                  = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.name.c_str();
        const char* co_name = cfg.co_name.c_str();

        TORCH_CHECK(hidden_dim % cfg.tile_N == 0,
                    "ASM kernel " + std::string(name) +
                        " is not supported for inter_dim = " + std::to_string(hidden_dim));

        auto result = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel " + kernelName);

    const auto& cfg = it->second;
    TORCH_CHECK(
        block_m == cfg.tile_M, __func__, " kernel: ", cfg.name, " need block_m == ", cfg.tile_M);

    KernelStage2Args kargs;
    kargs.o_buf          = out.data_ptr();
    kargs.a_buf          = input.data_ptr();
    kargs.tk_num_buf     = num_valid_ids.data_ptr();
    kargs.b_buf          = w2.data_ptr();
    kargs.as_buf         = a2_scale.value().data_ptr();
    kargs.bs_buf         = w2_scale.value().data_ptr();
    kargs.tk_buf         = sorted_token_ids.data_ptr();
    kargs.w_buf          = sorted_weights.value().data_ptr();
    kargs.expt_buf       = sorted_expert_ids.data_ptr();
    kargs.inter_dim      = hidden_dim;
    kargs.model_dim      = model_dim;
    kargs.tokens         = token_cnt;
    kargs.experts        = eprt;
    kargs.stride_a       = stride_a;
    kargs.stride_b       = stride_b;
    kargs.stride_o       = stride_o;
    kargs.stride_expt_b  = stride_exprt_b;
    kargs.stride_expt_bs = stride_exprt_b_scale;
    kargs.topks          = topk;
    kargs.split_k        = ksplit == 0 ? 1 : ksplit;
    kargs.o_flat_buf     = nullptr;
    kargs.dbg_i32        = nullptr;
    kargs.dbg_f32        = nullptr;
    kargs.dbg_f16        = nullptr;
    kargs.dbg_lds        = nullptr;

    int bdx         = 256;
    int gdx         = ((model_dim + cfg.tile_N - 1) / cfg.tile_N);
    int gdy         = sub_X_cnt;
    int gdz         = kargs.split_k;
    size_t arg_size = sizeof(kargs);

    // std::cout << "out:" << kargs.o_buf << std::endl;
    // std::cout << "a:" << kargs.a_buf << std::endl;
    // std::cout << "b:" << kargs.b_buf << std::endl;
    // std::cout << "tk_num_buf:" << kargs.tk_num_buf << std::endl;
    // std::cout << "as_buf:" << kargs.as_buf << std::endl;
    // std::cout << "bs_buf:" << kargs.bs_buf << std::endl;
    // std::cout << "tk_buf:" << kargs.tk_buf << std::endl;
    // std::cout << "w_buf:" << kargs.w_buf << std::endl;
    // std::cout << "expt_buf:" << kargs.expt_buf << std::endl;
    // std::cout << "inter_dim:" << kargs.inter_dim << std::endl;
    // std::cout << "model_dim:" << kargs.model_dim << std::endl;
    // std::cout << "tokens:" << kargs.tokens << std::endl;
    // std::cout << "experts:" << kargs.experts << std::endl;
    // std::cout << "stride_a:" << kargs.stride_a << std::endl;
    // std::cout << "stride_b:" << kargs.stride_b << std::endl;
    // std::cout << "stride_o:" << kargs.stride_o << std::endl;
    // std::cout << "stride_expt_b:" << kargs.stride_expt_b << std::endl;
    // std::cout << "stride_expt_bs:" << kargs.stride_expt_bs << std::endl;
    // std::cout << "topks:" << kargs.topks << std::endl;
    // std::cout << "arg_size:" << arg_size << std::endl;
    // printf("gdx:%d, gdy:%d, gdz:%d, tgs:%d\n", gdx, gdy, gdz, sub_X_cnt * gdx * gdz);

    impl_ptr->launch_kernel({&kargs,
                             &arg_size,
                             gdx, // gdx
                             gdy, // gdy
                             gdz, // gdz
                             bdx, // bdx: 4 wv64
                             1,   // bdy
                             1,   // bdz
                             stream});
}
