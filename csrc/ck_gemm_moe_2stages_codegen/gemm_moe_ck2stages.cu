// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "gemm_moe_ck2stages_lookup.h"
#include "gemm_moe_ck2stages.h"
#include "gemm_moe_ck2stages_heuristic_dispatch.hpp"
#include <cmath>

using MoeKernelMap = std::unordered_map<std::string, MoeKernel>;

// API for user aiter.ck_moe_stage1(...)

template <int stage = 1>
MoeKernel moe_dispatch(std::string &kernelName, int block_m)
{
    static const auto lookup = []
    {
        return MoeKernelMap{GENERATE_LOOKUP_TABLE()};
    }();

    if (kernelName != "")
    {
        auto it = lookup.find(kernelName);
        if (it != lookup.end())
        {
            auto kernel = it->second;
            return kernel;
        }
        std::cout << "[aiter] ck kernel not found: " << kernelName << std::endl;
    }
    if constexpr (stage == 1)
    {
        return moe_stage1_heuristic_dispatch(block_m);
    }
    else
    {
        return moe_stage2_heuristic_dispatch(block_m);
    }
}

void ck_moe_stage1(torch::Tensor &hidden_states,     // [m, k], input token
                   torch::Tensor &w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                   torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                   torch::Tensor &num_valid_ids,     // [1]
                   torch::Tensor &out,               // [m * topk, inter_dim]
                   int topk,
                   std::string &kernelName,
                   std::optional<torch::Tensor> w1_scale = std::nullopt, // [e, 1, n], gate(up) scale
                   std::optional<torch::Tensor> a1_scale = std::nullopt, // [m, 1], token scale
                   std::optional<int> block_m = 32,
                   std::optional<torch::Tensor> sorted_weights = std::nullopt)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
    at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(out.dtype() == at::ScalarType::BFloat16 || out.dtype() == at::ScalarType::Half,
                "Out dtype only support BFloat16/Float16!")

    int tokens = hidden_states.size(0);
    int sorted_size = sorted_token_ids.size(0);
    int E = w1.size(0);
    int N = w1.size(1) / 2;
    int K = hidden_states.size(-1);
    int MPerBlock = block_m.value();

    void *hidden_states_ptr = hidden_states.data_ptr();
    void *w1_ptr = w1.transpose(1, 2).data_ptr();
    void *w2_ptr = w2.data_ptr();
    void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
    void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
    void *num_valid_ids_ptr = num_valid_ids.data_ptr();
    void *sorted_weights_ptr = sorted_weights.has_value() ? sorted_weights.value().data_ptr() : nullptr;
    void *out_ptr = out.data_ptr();
    void *w1_scale_ptr = w1_scale.has_value() ? w1_scale.value().data_ptr() : nullptr;
    void *a1_scale_ptr = a1_scale.has_value() ? a1_scale.value().data_ptr() : nullptr;
    if (!hidden_states_ptr || !w1_ptr || !w2_ptr || !sorted_token_ids_ptr || !sorted_expert_ids_ptr || !num_valid_ids_ptr || !out_ptr)
    {
        std::cerr << "detect null ptr !" << std::endl;
        return;
    }

    auto kernel = moe_dispatch<1>(kernelName, MPerBlock);

    kernel(at::cuda::getCurrentCUDAStream().stream(),
           tokens, sorted_size, N, K, topk,
           hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);
}

void ck_moe_stage2(torch::Tensor &inter_states,      // [m, k], input token
                   torch::Tensor &w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                   torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                   torch::Tensor &num_valid_ids,     // [1]
                   torch::Tensor &out,               // [max_num_tokens_padded, inter_dim]
                   int topk,
                   std::string &kernelName,
                   std::optional<torch::Tensor> w2_scale = std::nullopt, // [e, 1, n], gate(up) scale
                   std::optional<torch::Tensor> a2_scale = std::nullopt, // [m, 1], token scale
                   std::optional<int> block_m = 32,
                   std::optional<torch::Tensor> sorted_weights = std::nullopt)
{
    TORCH_CHECK(out.dtype() == at::ScalarType::BFloat16 || out.dtype() == at::ScalarType::Half,
                "Out dtype only support BFloat16/Float16!")

    int tokens = inter_states.size(0);
    int sorted_size = sorted_token_ids.size(0);
    int E = w1.size(0);
    int N = w2.size(1);
    int K = inter_states.size(-1);
    int MPerBlock = block_m.value();

    void *inter_states_ptr = inter_states.data_ptr();
    void *w1_ptr = w1.data_ptr();
    void *w2_ptr = w2.data_ptr();
    void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
    void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
    void *sorted_weights_ptr = sorted_weights.has_value() ? sorted_weights.value().data_ptr() : nullptr;
    void *num_valid_ids_ptr = num_valid_ids.data_ptr();
    void *out_ptr = out.data_ptr();
    void *w2_scale_ptr = w2_scale.has_value() ? w2_scale.value().data_ptr() : nullptr;
    void *a2_scale_ptr = a2_scale.has_value() ? a2_scale.value().data_ptr() : nullptr;
    if (!inter_states_ptr || !w1_ptr || !w2_ptr || !sorted_token_ids_ptr || !sorted_expert_ids_ptr || !num_valid_ids_ptr || !out_ptr)
    {
        std::cerr << "detect null ptr !" << std::endl;
        return;
    }

    auto kernel = moe_dispatch<2>(kernelName, MPerBlock);

    kernel(at::cuda::getCurrentCUDAStream().stream(),
           tokens, sorted_size, N, K, topk,
           inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);
}