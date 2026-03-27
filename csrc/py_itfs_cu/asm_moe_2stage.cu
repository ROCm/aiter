// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <memory>
#include "aiter_hip_common.h"
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
    unsigned int total_tgs;
    p3 _p23;
    unsigned int ps_deno;
    p3 _p24;
    void *ptr_Qscl;
    p2 _p25;
    void *ptr_Qzero;
    p2 _p26;
    unsigned int eLQQs;
    p3 _p27;
};

struct __attribute__((packed)) Kernel2Args
{
    void *ptr_OBuffer;
    p2 _p0;
    void *ptr_XBuffer;
    p2 _p1;
    void *ptr_DBuffer;
    p2 _p2;
    void *ptr_XCBuffer;
    p2 _p3;
    void *ptr_ScaleXBuffer;
    p2 _p4;
    void *ptr_ScaleDBuffer;
    p2 _p5;
    void *ptr_STPBuffer;
    p2 _p6;
    void *ptr_SEPBuffer;
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
    void *ptr_SWBuffer;
    p2 _p19;
    void *ptr_DScaleBuffer;
    p2 _p20;
    void *ptr_DZeroBuffer;
    p2 _p21;
    unsigned int stride_expert_dequant_D;
    p3 _p22;
};

static CFG *get_cfg(aiter_tensor_t *inp, aiter_tensor_t *out, aiter_tensor_t *w1, QuantType quant_type, bool do_weight)
{
    bool is_MultiX = (inp->numel() / inp->size(-1)) == (out->numel() / out->size(-1));

    if (inp->dtype() == AITER_DTYPE_fp8 &&
        w1->dtype() == AITER_DTYPE_fp8 &&
        out->dtype() == AITER_DTYPE_bf16 &&
        quant_type == QuantType::per_Token &&
        do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_doweight_g1u1;
    }
    else if (inp->dtype() == AITER_DTYPE_fp8 &&
             w1->dtype() == AITER_DTYPE_fp8 &&
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
        if (is_MultiX)
            return &cfg_fmoe_stage1_bf16_pertokenInt8_g1u1_multix;
        return &cfg_fmoe_stage1_bf16_pertokenInt8_g1u1;
    }
    else if (inp->dtype() == AITER_DTYPE_fp8 &&
             w1->dtype() == AITER_DTYPE_fp8 &&
             out->dtype() == AITER_DTYPE_fp8 &&
             quant_type == QuantType::per_1x128 &&
             !do_weight)
    {
        return &cfg_fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1;
    }
    else
    {
        AITER_CHECK(false, __func__, " Unsupported input_type:", AiterDtype_to_str(inp->dtype()),
                    ", weight_type:", AiterDtype_to_str(w1->dtype()),
                    ", out_type:", AiterDtype_to_str(out->dtype()),
                    ", quant_type:", static_cast<int>(quant_type), ", do_weight:", do_weight);
        return nullptr;
    }
};

static CFG *get_cfg_stage2(aiter_tensor_t *inter_states, aiter_tensor_t *out, aiter_tensor_t *w2, QuantType quant_type, bool do_weight)
{
    if (inter_states->dtype() == AITER_DTYPE_i8 &&
        w2->dtype() == AITER_DTYPE_i8 &&
        out->dtype() == AITER_DTYPE_bf16 &&
        quant_type == QuantType::per_Token &&
        do_weight)
    {
        return &cfg_fmoe_stage2_bf16_pertokenInt8_g1u1;
    }
    else
    {
        AITER_CHECK(false, __func__, " Unsupported input_type:", AiterDtype_to_str(inter_states->dtype()),
                    ", weight_type:", AiterDtype_to_str(w2->dtype()),
                    ", out_type:", AiterDtype_to_str(out->dtype()),
                    ", quant_type:", static_cast<int>(quant_type), ", do_weight:", do_weight);
        return nullptr;
    }
}

static std::string get_heuristic_kernel(int m_num, int N, int blockk_size, CFG *cfgs, std::string arch_id)
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
            uint32_t cand_empty = local_round * num_cu - tg_num;
            if (empty_cu > cand_empty)
            {
                round = local_round;
                selected = el.first;
                empty_cu = cand_empty;
            }
            else if (empty_cu == cand_empty)
            {
                // Tie-break: prefer non-buffer kernel when both have same tile
                bool cur_is_buffer = (selected.find("_buffer") != std::string::npos);
                bool cand_is_buffer = (el.first.find("_buffer") != std::string::npos);
                if (!cand_is_buffer && cur_is_buffer)
                    selected = el.first;
            }
        }
    }
    return selected;
}

AITER_C_ITFS void moe_stage1_g1u1(
    aiter_tensor_t *input,             // [token_cnt, model_dim] M,K
    aiter_tensor_t *w1,                // [expert, inter_dim*2, model_dim] N,K
    aiter_tensor_t *w2,                // [expert, model_dim, inter_dim]
    aiter_tensor_t *sorted_token_ids,  // [max_num_tokens_padded]
    aiter_tensor_t *sorted_expert_ids, // [max_num_m_blocks]
    aiter_tensor_t *num_valid_ids,     // [1]
    aiter_tensor_t *out,               // [token_cnt, topk, inter_dim*2]
    int inter_dim,
    const char *kernelName,
    int block_m,
    int ksplit,
    int activation,
    int quant_type,
    aiter_tensor_t *a1_scale,       // [token_cnt, 1], token scale
    aiter_tensor_t *w1_scale,       // [expert, 1, inter_dim], gate(up) scale
    aiter_tensor_t *w1_lqq_scale,   // [expert, inter_dim*2, model_dim/group_in_k_lqq] N,Klqq
    aiter_tensor_t *w1_lqq_zero,    // [expert, inter_dim*2, model_dim/group_in_k_lqq] N,Klqq
    aiter_tensor_t *fc2_smooth_scale, // [expert, 1, inter_dim], smooth quant scale
    aiter_tensor_t *fc2_scale,       // [expert, 1, dim], fc2 scale
    aiter_tensor_t *sorted_weights, // [max_num_tokens_padded], do_weight==true need
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(input->device_id);
    ActivationType act = static_cast<ActivationType>(activation);
    QuantType qt = static_cast<QuantType>(quant_type);

    CFG *config_map = get_cfg(input, out, w1, qt, sorted_weights != nullptr);
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    int model_dim = input->size(-1);
    int hidden_dim = inter_dim;

    int model_dim_w1 = w1->size(2);
    int model_dim_w2 = w2->size(1);
    int gu_int4 = (model_dim_w2 / model_dim_w1 == 2) ? 1 : 0;

    int token_cnt = out->size(0);
    int topk = out->size(1);
    int eprt = w1->size(0);

    // sub_X -> block_m, batch -> token_cnt, sz_sep -> gdy
    long long sz_stp = (long long)topk * token_cnt + (long long)eprt * block_m - topk;
    sz_stp = (sz_stp + block_m - 1) / block_m;
    int sub_X_cnt = sorted_expert_ids->size(0);

    std::string arch_id = get_gpu_arch();
    std::string kernelNameStr = (kernelName && kernelName[0] != '\0') ? arch_id + kernelName : "";
    if (kernelNameStr.empty())
    {
        kernelNameStr = get_heuristic_kernel(sub_X_cnt, inter_dim, block_m, config_map, arch_id);
    }

    AiterAsmKernel *impl_ptr = nullptr;
    auto it = config_map->find(kernelNameStr);
    if (it != config_map->end())
    {
        const auto &cfg = it->second;
        const char *name = cfg.knl_name.c_str();
        const char *co_name = cfg.co_name.c_str();

        AITER_CHECK(inter_dim % cfg.tile_n == 0,
                    "ASM kernel " + std::string(name) + " is not supported for inter_dim = " + std::to_string(inter_dim));

        auto result = impl_ptr_map.emplace(name, nullptr);
        if (result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        AITER_CHECK(false, __func__, " not find kernel " + kernelNameStr);

    const auto &cfg = it->second;
    uint32_t sub_GU = cfg.tile_n;
    AITER_CHECK(block_m == cfg.tile_m, __func__, " kernel: ", cfg.knl_name, " need block_m == ", cfg.tile_m);

    int stride_X = input->stride(-2) * input->element_size();
    int stride_GU = model_dim * w1->element_size();

    int stride_expert_GU = stride_GU * inter_dim;
    int stride_expert_GUDQN = w1_scale ? w1_scale->stride(0) * sizeof(float) : 0;
    int stride_expert_SMTDQN = inter_dim * sizeof(float);
    int stride_O = topk * inter_dim * out->element_size();
    int stride_expert_LQQ = w1_lqq_scale ? w1_lqq_scale->stride(0) : 0;

    if (gu_int4)
    {
        stride_GU /= 2;
        stride_expert_GU /= 2;
    }

    if (inter_dim * 2 == w1->size(1))
    {
        stride_expert_GU *= 2;
    }

    // Determine if this kernel uses the extended args layout (multix/lqq kernels)
    // Old .co files expect the legacy arg size (up to ptr_SW + padding).
    bool use_extended_args = (config_map == &cfg_fmoe_stage1_bf16_pertokenInt8_g1u1_multix);
    static constexpr size_t legacy_arg_size = offsetof(KernelArgs, total_tgs);

    KernelArgs args;
    size_t arg_size = use_extended_args ? sizeof(args) : legacy_arg_size;
    memset(&args, 0, sizeof(args));
    args.ptr_O = out->ptr;
    args.ptr_X = input->ptr;
    args.ptr_GU = w1->ptr;
    args.ptr_XC = num_valid_ids->ptr;

    args.ptr_XQ = a1_scale ? a1_scale->ptr : nullptr;
    args.ptr_GUQ = w1_scale ? w1_scale->ptr : nullptr;
    args.ptr_SMQ = fc2_smooth_scale ? fc2_smooth_scale->ptr : nullptr;

    args.ptr_STP = sorted_token_ids->ptr;
    args.ptr_SEP = sorted_expert_ids->ptr;
    args.dim = model_dim;
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
    args.ptr_SW = sorted_weights ? sorted_weights->ptr : nullptr;
    args.total_tgs = 0;
    args.ps_deno = ((inter_dim + sub_GU - 1) / sub_GU);
    args.ptr_Qscl = w1_lqq_scale ? w1_lqq_scale->ptr : nullptr;
    args.ptr_Qzero = w1_lqq_zero ? w1_lqq_zero->ptr : nullptr;
    args.eLQQs = stride_expert_LQQ;

    uint32_t k_num = 1 << ksplit;
    AITER_CHECK(model_dim % k_num == 0, __func__, " Unsupported ksplit for model_dim:", model_dim, " k_num:", k_num);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};

    int bdx = 256;
    int gdx = ((hidden_dim + sub_GU - 1) / sub_GU);
    int gdy = use_extended_args ? sz_stp : sub_X_cnt;
    int gdz = k_num;

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

AITER_C_ITFS void moe_stage2_g1u1(
    aiter_tensor_t *inter_states,      // [token_cnt, topk, inter_dim]
    aiter_tensor_t *w1,                // [expert, inter_dim*2, model_dim] N,K
    aiter_tensor_t *w2,                // [expert, model_dim, inter_dim]
    aiter_tensor_t *sorted_token_ids,  // [max_num_tokens_padded]
    aiter_tensor_t *sorted_expert_ids, // [max_num_m_blocks]
    aiter_tensor_t *num_valid_ids,     // [1]
    aiter_tensor_t *out,               // [token_cnt, topk, model_dim]
    int topk,
    const char *kernelName,
    int block_m,
    aiter_tensor_t *w2_scale,       // [expert, 1, dim], down scale
    aiter_tensor_t *a2_scale,       // [token_cnt, 1], inter scale
    aiter_tensor_t *w2_lqq_scale,   // [expert, inter_dim/group_in_k_lqq, model_dim] N,Klqq
    aiter_tensor_t *w2_lqq_zero,    // [expert, inter_dim/group_in_k_lqq, model_dim] N,Klqq
    aiter_tensor_t *sorted_weights, // [max_num_tokens_padded], do_weight==true need
    int quant_type,
    int activation,
    int splitk,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(inter_states->device_id);
    QuantType qt = static_cast<QuantType>(quant_type);

    CFG *config_map = get_cfg_stage2(inter_states, out, w2, qt, sorted_weights != nullptr);
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;

    int inter_dim = inter_states->size(-1);

    int model_dim_w1 = w1->size(2);
    int model_dim_w2 = w2->size(1);
    int isInt4 = (model_dim_w2 / model_dim_w1 == 2) ? 1 : 0;

    int sub_X_cnt = sorted_expert_ids->size(0);
    std::string arch_id = get_gpu_arch();
    std::string kernelNameStr = (kernelName && kernelName[0] != '\0') ? arch_id + kernelName : "";
    if (kernelNameStr.empty())
    {
        kernelNameStr = get_heuristic_kernel(sub_X_cnt, out->size(-1), block_m, config_map, arch_id);
    }

    AiterAsmKernel *impl_ptr = nullptr;
    auto it = config_map->find(kernelNameStr);
    if (it != config_map->end())
    {
        const auto &cfg = it->second;
        const char *name = cfg.knl_name.c_str();
        const char *co_name = cfg.co_name.c_str();

        auto result = impl_ptr_map.emplace(name, nullptr);
        if (result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        AITER_CHECK(false, __func__, " not find kernel " + kernelNameStr);

    int token_cnt = inter_states->size(0);

    int dim = out->size(-1);
    int eprt = w2->size(0);
    const auto &cfg = it->second;
    uint32_t sub_D = cfg.tile_n;
    AITER_CHECK(block_m == cfg.tile_m, __func__, " kernel: ", cfg.knl_name, " need block_m == ", cfg.tile_m);

    int stride_X = inter_dim * inter_states->element_size();
    int stride_D = inter_dim * w2->element_size();
    int stride_expert_D = inter_dim * dim * w2->element_size();
    int stride_expert_scale_D = w2_scale ? dim * sizeof(float) : 0;
    int stride_expert_dequant_D = w2_lqq_scale ? w2_lqq_scale->stride(0) : 0;

    if (isInt4)
    {
        stride_D /= 2;
        stride_expert_D /= 2;
    }

    int dbl_o = (splitk > 1) ? 2 : 1;
    int stride_O = dim * out->element_size() * dbl_o;

    Kernel2Args args;
    size_t arg_size = sizeof(args);
    memset(&args, 0, sizeof(args));
    args.ptr_OBuffer = out->ptr;
    args.ptr_XBuffer = inter_states->ptr;
    args.ptr_DBuffer = w2->ptr;
    args.ptr_XCBuffer = num_valid_ids->ptr;

    args.ptr_ScaleXBuffer = a2_scale ? a2_scale->ptr : nullptr;
    args.ptr_ScaleDBuffer = w2_scale ? w2_scale->ptr : nullptr;

    args.ptr_STPBuffer = sorted_token_ids->ptr;
    args.ptr_SEPBuffer = sorted_expert_ids->ptr;
    args.dim = dim;
    args.hidden_dim = inter_dim;
    args.token_cnt = token_cnt;
    args.eprt_cnt = eprt;
    args.stride_X = stride_X;
    args.stride_D = stride_D;
    args.stride_O = stride_O;
    args.stride_expert_D = stride_expert_D;
    args.stride_expert_scale_D = stride_expert_scale_D;
    args.topk = topk;
    args.splitk = splitk;
    args.ptr_SWBuffer = sorted_weights ? sorted_weights->ptr : nullptr;
    args.stride_expert_dequant_D = stride_expert_dequant_D;
    args.ptr_DScaleBuffer = w2_lqq_scale ? w2_lqq_scale->ptr : nullptr;
    args.ptr_DZeroBuffer = w2_lqq_zero ? w2_lqq_zero->ptr : nullptr;

    uint32_t k_num = 1;

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};

    int bdx = 256;
    int gdx = ((dim + sub_D - 1) / sub_D);
    int gdy = sub_X_cnt;
    int gdz = k_num;

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
