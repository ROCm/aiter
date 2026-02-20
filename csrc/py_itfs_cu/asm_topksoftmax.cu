// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>
#include "asm_topksoftmax_configs.hpp"

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

const CFG::Entry* get_heuristic_kernel_topksoftmax(GPUArchId arch_id,
    std::string dtype,
    int subm,
    int num_experts,
    int topk,
    const CFG* cfgs)
{
    for(const auto& cfg : cfgs->get_configs_for_arch(arch_id))
    {
        if (cfg.dtype != dtype || cfg.subm != subm || cfg.num_experts != num_experts || cfg.topk != topk)
            continue;
        return &cfg;
    }

    TORCH_CHECK(false,
    __func__,
    ": cannot get heuristic kernel!"
    " arch_id:", arch_id,
    " dtype:", dtype,
    " subm:", subm,
    " num_experts:", num_experts,
    " topk:", topk);
    return nullptr;
}

void topk_softmax_asm(torch::Tensor& topk_weights,         // [num_tokens, topk]
                      torch::Tensor& topk_indices,         // [num_tokens, topk]
                      torch::Tensor& token_expert_indices, // [num_tokens, topk]
                      torch::Tensor& gating_output,        // [num_tokens, num_experts]
                      bool need_renorm)
{
    auto arch_id           = get_gpu_arch();
    const uint num_experts = gating_output.size(-1);
    const uint num_tokens  = gating_output.numel() / num_experts;
    const uint topk        = topk_weights.size(-1);
    const uint out_stride  = topk_weights.stride(0);
    const uint SUBM        = num_tokens < 10000 ? 4 : 12;
    std::string dtype;
    if(gating_output.dtype() == at::ScalarType::Float)
        dtype = "fp32";
    else if(gating_output.dtype() == at::ScalarType::BFloat16)
        dtype = "bf16";
    else
        TORCH_CHECK(false, __func__, ": unsupport gating_output dtype:", gating_output.scalar_type());

    KernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_T      = (void*)topk_indices.data_ptr();
    args.ptr_W      = (void*)topk_weights.data_ptr();
    args.ptr_A      = (void*)gating_output.data_ptr();

    args.batch       = num_tokens;
    args.expert      = num_experts;
    args.topk        = topk;
    args.renormalize = need_renorm ? 1 : 0;
    args.out_stride  = out_stride * 4;

    const CFG* config_map = &cfg_topksoftmax;

    const auto* cfg = get_heuristic_kernel_topksoftmax(arch_id, dtype, SUBM, num_experts, topk, config_map);

    AiterAsmKernel<>* impl_ptr = config_map->load_kernel_for_config(cfg);

    TORCH_CHECK(impl_ptr != nullptr, __func__, " not find kernel");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    uint gdx = (num_tokens + SUBM - 1) / SUBM;
    TORCH_CHECK(gdx >> 31 == 0, "num_tokens too large: ", num_tokens);
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