// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <climits>
#include <tuple>

namespace aiter::moe_gfx1250
{
constexpr int PERSISTENT_TG = 256;
constexpr int THREADS       = 128;

inline bool is_fp4(AiterDtype dtype)
{
    return dtype == AITER_DTYPE_fp4x2 || dtype == AITER_DTYPE_u8;
}

inline bool is_scale(AiterDtype dtype)
{
    return dtype == AITER_DTYPE_fp8_e8m0 || dtype == AITER_DTYPE_u8;
}

inline void check_tensor(const aiter_tensor_t* tensor, const char* name, int ndim, int device)
{
    AITER_CHECK(tensor != nullptr, name, " must not be null");
    AITER_CHECK(tensor->is_gpu() && tensor->device_id == device,
                name, " must be on the same GPU as the activation");
    AITER_CHECK(tensor->dim() == ndim, name, " must have ", ndim, " dimensions");
    AITER_CHECK(tensor->is_contiguous(), name, " must be contiguous");
}

inline void check_shape(const aiter_tensor_t* tensor,
                        std::initializer_list<int64_t> shape,
                        const char* name)
{
    AITER_CHECK(tensor->dim() == static_cast<int>(shape.size()), name, " has wrong rank");
    int axis = 0;
    for(int64_t size : shape)
    {
        AITER_CHECK(tensor->size(axis) == size,
                    name, " dimension ", axis, " must be ", size,
                    ", got ", tensor->size(axis));
        ++axis;
    }
}

inline int dimension(int64_t value, const char* name)
{
    AITER_CHECK(value > 0 && value <= INT_MAX, name, " must be in [1, INT_MAX]");
    return static_cast<int>(value);
}

inline size_t scale_rows(size_t capacity)
{
    return (capacity + 31) / 32 * 32;
}

inline void check_routing(const aiter_tensor_t* sorted_token_ids,
                          const aiter_tensor_t* sorted_expert_ids,
                          const aiter_tensor_t* num_valid_ids,
                          int64_t tokens,
                          int64_t topk,
                          int64_t block_m,
                          int device)
{
    AITER_CHECK(tokens >= 0 && tokens < (1 << 24), "num_tokens must fit in 24 bits");
    AITER_CHECK(topk > 0 && topk < 256, "topk must be in [1, 255]");
    AITER_CHECK(block_m == 16 || block_m == 32 || block_m == 64,
                "block_m must be 16, 32 or 64, matching the sorting block size");
    for(const auto& [tensor, name] :
        {std::pair{sorted_token_ids, "sorted_token_ids"},
         std::pair{sorted_expert_ids, "sorted_expert_ids"},
         std::pair{num_valid_ids, "num_valid_ids"}})
    {
        check_tensor(tensor, name, 1, device);
        AITER_CHECK(tensor->dtype() == AITER_DTYPE_i32, name, " must be int32");
    }
    AITER_CHECK(num_valid_ids->numel() >= 1, "num_valid_ids must contain the padded row count");
    AITER_CHECK(sorted_token_ids->numel() <= INT_MAX, "sorted_token_ids capacity is too large");
    AITER_CHECK(sorted_expert_ids->numel() >=
                    (sorted_token_ids->numel() + block_m - 1) / block_m,
                "sorted_expert_ids must hold one expert per sorting block");
}

inline void check_scales(const aiter_tensor_t* a_scale,
                         const aiter_tensor_t* b_scale,
                         int experts,
                         int n,
                         int k,
                         size_t sorted_capacity,
                         int device)
{
    check_tensor(a_scale, "a_scale", 2, device);
    check_tensor(b_scale, "w_scale", 3, device);
    AITER_CHECK(is_scale(a_scale->dtype()) && is_scale(b_scale->dtype()),
                "scales must contain E8M0 bytes (float8_e8m0fnu or uint8)");
    AITER_CHECK(k % 128 == 0, "K must be divisible by 128 for the scale shuffle");
    AITER_CHECK(a_scale->size(1) == k / 32 &&
                    static_cast<size_t>(a_scale->size(0)) >= scale_rows(sorted_capacity),
                "a_scale must have shape [at least round_up(sorted capacity,32), K/32]");
    check_shape(b_scale, {experts, n, k / 32}, "w_scale");
}

template <typename Config>
inline bool supports(const Config& cfg, int block_m, int n, int k)
{
    return cfg.tile_m == block_m && cfg.tile_n > 0 && cfg.tile_k > 0 &&
           cfg.pf_tiles > 0 && cfg.cluster_y == 1 &&
           (cfg.cluster_x == 1 || cfg.cluster_x == 4) &&
           n % (cfg.tile_n * cfg.cluster_x) == 0 &&
           k % cfg.tile_k == 0 && k / cfg.tile_k >= cfg.pf_tiles;
}

template <typename ConfigMap>
const typename ConfigMap::mapped_type& select_config(const ConfigMap& configs,
                                                      const char* a_intype,
                                                      const char* outtype,
                                                      int block_m,
                                                      int n,
                                                      int k,
                                                      const char* kernel_name)
{
    const std::string arch = get_gpu_arch();
    AITER_CHECK(arch == "gfx1250", "MoE ASM requires gfx1250, got ", arch);
    AITER_CHECK(!configs.empty(), "no MoE ASM kernels registered; build with AITER_GPU_ARCHS=gfx1250");
    if(kernel_name != nullptr && kernel_name[0] != '\0')
    {
        const auto found = configs.find(arch + kernel_name);
        AITER_CHECK(found != configs.end(), "unregistered MoE kernelName: ", kernel_name);
        const auto& cfg = found->second;
        AITER_CHECK(cfg.a_intype == a_intype && cfg.outtype == outtype,
                    "kernelName does not match activation/output dtype: ", kernel_name);
        AITER_CHECK(supports(cfg, block_m, n, k),
                    "kernelName does not support block_m/N/K=", block_m, "/", n, "/", k,
                    "; requires matching tile_m, N%(tile_n*cluster_x)==0, "
                    "K%tile_k==0 and K>=tile_k*pf_tiles");
        return cfg;
    }

    const typename ConfigMap::mapped_type* best = nullptr;
    const auto rank = [](const auto& cfg) {
        return std::tuple{cfg.priority, -cfg.cluster_x, -cfg.tile_n,
                          std::string_view(cfg.knl_name)};
    };
    for(const auto& [name, cfg] : configs)
    {
        if(cfg.arch != arch || cfg.a_intype != a_intype || cfg.outtype != outtype ||
           !supports(cfg, block_m, n, k))
            continue;
        // Prefer base variants, then complete 4x1 clusters, then N512.
        // The name breaks ties independently of unordered_map iteration order.
        if(best == nullptr || rank(cfg) < rank(*best))
            best = &cfg;
    }
    AITER_CHECK(best != nullptr,
                "no compatible MoE ASM kernel for ", a_intype, " -> ", outtype,
                ", block_m/N/K=", block_m, "/", n, "/", k,
                "; requires N divisible by 384 or 512 and a complete K prefetch window");
    return *best;
}

template <typename Config, typename Args>
void launch(const Config& cfg, Args& args, size_t arg_size, hipStream_t stream)
{
    static SynchronizedCache<std::string_view, std::unique_ptr<AiterAsmKernel>> kernels;
    auto& kernel = kernels.get_or_create(cfg.knl_name, [&]() {
        return std::make_unique<AiterAsmKernel>(cfg.knl_name.c_str(), cfg.co_name.c_str());
    });
    // GRID_X/Y are compiled into the shader's persistent tile walk. As in the
    // host, launch all 256 workgroups on X, with the selected cluster size.
    kernel->launch_kernel({&args, &arg_size, PERSISTENT_TG, 1, 1, THREADS, 1, 1,
                           stream, cfg.cluster_x, cfg.cluster_y, 1});
}
} // namespace aiter::moe_gfx1250
