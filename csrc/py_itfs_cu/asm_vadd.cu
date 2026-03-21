// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_vadd_configs.hpp"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <memory>
#include <torch/all.h>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_a;
    p2 _p0;
    void* ptr_b;
    p2 _p1;
    void* ptr_c;
    p2 _p2;
    unsigned int width;
    p3 _p3;
    unsigned int height;
    p3 _p4;
};

void vadd_asm(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c)
{
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat && c.scalar_type() == at::kFloat,
                __func__,
                ": expect fp32 tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, __func__, ": expect 2-D tensors");
    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(), __func__, ": shape mismatch");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), __func__, ": expect contiguous");

    const int64_t height = a.size(0);
    const int64_t width  = a.size(1);
    TORCH_CHECK(width > 0 && height > 0, __func__, ": empty tensor");
    TORCH_CHECK(width <= 0x7fffffff && height <= 0x7fffffff, __func__, ": tensor too large");

    std::string arch_id = get_gpu_arch();
    const vaddConfig* cfg = nullptr;
    for(const auto& el : cfg_vadd)
    {
        if(el.second.arch == arch_id)
        {
            cfg = &el.second;
            break;
        }
    }
    TORCH_CHECK(cfg != nullptr, __func__, ": no vadd asm kernel for arch ", arch_id);

    KernelArgs args{};
    size_t arg_size = sizeof(args);
    args.ptr_a  = a.data_ptr();
    args.ptr_b  = b.data_ptr();
    args.ptr_c  = c.data_ptr();
    args.width  = static_cast<unsigned int>(width);
    args.height = static_cast<unsigned int>(height);

    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    const char* name    = cfg->knl_name.c_str();
    const char* co_name = cfg->co_name.c_str();
    auto result         = impl_ptr_map.emplace(name, nullptr);
    if(result.second)
    {
        result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
    }
    AiterAsmKernel* impl_ptr = result.first->second.get();

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(a));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    constexpr int tbx = 16;
    constexpr int tby = 16;
    const int gdx       = static_cast<int>((width + tbx - 1) / tbx);
    const int gdy       = static_cast<int>((height + tby - 1) / tby);
    TORCH_CHECK(gdx > 0 && gdy > 0, __func__, ": bad grid");
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,
                             gdy,
                             1,
                             tbx,
                             tby,
                             1,
                             stream});
}
