// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_topksoftmax_configs.hpp"
#include <memory>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_T;
    p2 _p0;
    void* ptr_W;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    unsigned int batch;
    p3 _p4;
    unsigned int expert;
    p3 _p5;
    unsigned int topk;
    p3 _p6;
    unsigned int renormalize;
    p3 _p7;
    unsigned int out_stride;
    p3 _p8;
};

std::pair<std::string, int> get_heuristic_kernel_topksoftmax(std::string arch_id,
    std::string dtype,
    int max_subm,
    int num_experts,
    int topk,
    CFG* cfgs)
{
    int subm = -1;
    std::string kernelName = "";
    for(const auto& el : *cfgs)
    {
        if (el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if (cfg.dtype != dtype || cfg.subm > max_subm || cfg.num_experts != num_experts || cfg.topk != topk)
            continue;
        
        if (cfg.subm > subm)
        {
            kernelName = el.first;
            subm = cfg.subm;
        }
    }

    AITER_CHECK(!kernelName.empty(),
    __func__,
    ": cannot get heuristic kernel!"
    " arch_id:", arch_id,
    " dtype:", dtype,
    " max_subm:", max_subm,
    " num_experts:", num_experts,
    " topk:", topk);
    return {kernelName, subm};
}

AITER_C_ITFS
void topk_softmax_asm(aiter_tensor_t* topk_weights,         // [num_tokens, topk]
                      aiter_tensor_t* topk_indices,         // [num_tokens, topk]
                      aiter_tensor_t* token_expert_indices, // [num_tokens, topk]
                      aiter_tensor_t* gating_output,        // [num_tokens, num_experts]
                      int need_renorm, hipStream_t stream)
{
    std::string arch_id = get_gpu_arch();
    const uint num_experts = gating_output->size(-1);
    const uint num_tokens  = gating_output->numel() / num_experts;
    const uint topk        = topk_weights->size(-1);
    const uint out_stride  = topk_weights->stride(0);
    const uint MAX_SUBM    = num_tokens < 10000 ? 4 : 12;
    std::string dtype;
    if(gating_output->dtype() == AITER_DTYPE_fp32)
        dtype = "fp32";
    else if(gating_output->dtype() == AITER_DTYPE_bf16)
        dtype = "bf16";
    else
        AITER_CHECK(false, __func__, ": unsupport gating_output dtype:", AiterDtype_to_str(gating_output->dtype()));

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_T      = topk_indices->ptr;
    args.ptr_W      = topk_weights->ptr;
    args.ptr_A      = gating_output->ptr;

    args.batch       = num_tokens;
    args.expert      = num_experts;
    args.topk        = topk;
    args.renormalize = need_renorm ? 1 : 0;
    args.out_stride  = out_stride * 4;

    CFG* config_map = &cfg_topksoftmax;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    AiterAsmKernel* impl_ptr = nullptr;
    auto [kernelName, subm] = get_heuristic_kernel_topksoftmax(arch_id, dtype, MAX_SUBM, num_experts, topk, config_map);

    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        auto result         = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }
        impl_ptr = result.first->second.get();
    }
    else
        AITER_CHECK(false, __func__, " not find kernel " + kernelName);


    const HipDeviceGuard device_guard(gating_output->device_id);

    uint gdx = (num_tokens + subm - 1) / subm;
    AITER_CHECK(gdx >> 31 == 0, "num_tokens too large: ", num_tokens);
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             static_cast<int>(gdx), // gdx
                             1,                     // gdy
                             1,                     // gdz
                             256,                   // bdx: 4 wv64
                             1,                     // bdy
                             1,                     // bdz
                             stream});
}