// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "moe_op.h"
#include "gemm_moe_ck2stages_manifest.h"
#include "gemm_moe_ck2stages_lookup.h"
#include "gemm_moe_ck2stages.h"
#include <cmath>

using MoeKernel = std::function<
    void(const hipStream_t &stream, int, int, int, int,
         int,
         void *&,
         void *&,
         void *&,
         void *&,
         void *&,
         void *&,
         void *&,
         void *&,
         std::optional<void *>,
         std::optional<void *>)>;

using MoeKernelMap = std::unordered_map<std::string, MoeKernel>;

// API for user aiter.ck_moe_stage1(...)

void ck_moe_stage1(torch::Tensor &hidden_states,     // [m, k], input token
    torch::Tensor &w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
    torch::Tensor &w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
    torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
    torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
    torch::Tensor &num_valid_ids,     // [1]
    torch::Tensor &out,               // [m * topk, inter_dim]
    int topk,
    std::string &kernelName,
    std::optional<torch::Tensor> w1_scale        = std::nullopt, // [e, 1, n], gate(up) scale
    std::optional<torch::Tensor> a1_scale        = std::nullopt, // [m, 1], token scale
    std::optional<int> block_m                   = 32,
    std::optional<torch::Tensor> sorted_weights  = std::nullopt,
    std::optional<int> act_op                    = 0,
    std::optional<int> pipe_ver                  = 1)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
    at::cuda::getCurrentCUDAStream().stream();
    // TORCH_CHECK(hidden_states.dtype() == w1.dtype(),
    //             "Weights and activations should both be same dtype!");

    TORCH_CHECK(out.dtype() == at::ScalarType::BFloat16 || out.dtype() == at::ScalarType::Half,
                "Out dtype only support BFloat16/Float16!")

    int tokens = hidden_states.size(0);
    int sorted_size = sorted_token_ids.size(0);
    int E = w1.size(0);
    int N = w1.size(1);
    int K = hidden_states.size(-1);
    // int max_num_tokens_padded = sorted_token_ids.size(0);
    // int agvtokens_per_expert = max_num_tokens_padded / E;
    int MPerBlock = block_m.value();
    int ACTOP = act_op.value();
    bool isPerTensorQuant = (!w1_scale.has_value()) || (w1_scale.value().numel() == E);

    // int M = agvtokens_per_expert < 32 ? 32 : (agvtokens_per_expert < 64 ? 64 : 128);
    QuantType q_type = isPerTensorQuant ? QuantType::per_Tensor : QuantType::per_Token;
    void *hidden_states_ptr = hidden_states.data_ptr();
    void *w1_ptr = w1.transpose(1, 2).data_ptr();
    void *w2_ptr = w2.data_ptr();
    void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
    void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
    void *num_valid_ids_ptr = num_valid_ids.data_ptr();
    void *sorted_weights_ptr = sorted_weights.has_value() ? sorted_weights.value().data_ptr() : nullptr;
    void *out_ptr = out.data_ptr();
    void *w1_scale_ptr = w1_scale.has_value() ? w1_scale.value().transpose(0, 1).data_ptr() : nullptr;
    void *a1_scale_ptr = a1_scale.has_value() ? a1_scale.value().data_ptr() : nullptr;
    if (!hidden_states_ptr || !w1_ptr || !w2_ptr || !sorted_token_ids_ptr || !sorted_expert_ids_ptr || !num_valid_ids_ptr || !out_ptr)
    {
        std::cerr << "detect null ptr ！" << std::endl;
        return;
    }

    static const auto lookup = []
    {
        return MoeKernelMap{GENERATE_LOOKUP_TABLE()};
    }();

    auto it = lookup.find(kernelName);

    // If we found an optimal kernel, use it.
    if (it != lookup.end())
    {
        std::cout << "[aiter] found CK kernel : " << kernelName << std::endl;
        return it->second;
    }
    //std::cerr << "[aiter] ck kernel not found: " << kernelName << std::endl;
    // Otherwise, use heuristics.
    
    // BF16
    if (inter_states.dtype() == at::ScalarType::BFloat16)
    {
        if(q_type == QuantType::per_Tensor)
        {
            if (sorted_weights.has_value())
            {
                if(ACTOP == 0)
                {
                    if (MPerBlock == 32) 
                    {
                        using A0DataType = B16;
                        using B0DataType = B16;
                        using AccDataType = F32;
                        using EDataType = B16;
                        const bool Nswizzle = false;
                        using CDEElementOp = TypeCastExpertWeight;
                        if (K % (256 / sizeof(A0DataType)) == 0)
                        {
                            ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 1, 256, 32, 128, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, true, 0>
                            (at::cuda::getCurrentCUDAStream().stream(), 
                            tokens, sorted_size, N, K, topk, 
                            hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);

                        }
                    }
                }
            }
        }
    }
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
    std::optional<torch::Tensor> sorted_weights = std::nullopt, // [max_num_tokens_padded])
    std::optional<int> pipe_ver                 = 1)
{
    // TORCH_CHECK(inter_states.dtype() == w2.dtype(),
    //             "Weights and activations should both be same dtype!");
    //
    TORCH_CHECK(out.dtype() == at::ScalarType::BFloat16 || out.dtype() == at::ScalarType::Half,
                "Out dtype only support BFloat16/Float16!")

    int tokens = inter_states.size(0);
    int sorted_size = sorted_token_ids.size(0);
    int E = w1.size(0);
    int N = w2.size(1);
    int K = inter_states.size(-1);
    // int max_num_tokens_padded = sorted_token_ids.size(0);
    // int agvtokens_per_expert = max_num_tokens_padded / E;
    int MPerBlock = block_m.value();
    // int M = agvtokens_per_expert < 32 ? 32 : (agvtokens_per_expert < 64 ? 64 : 128);
    bool isPerTensorQuant = (!w2_scale.has_value()) || (w2_scale.value().numel() == E);
    QuantType q_type = isPerTensorQuant ? QuantType::per_Tensor : QuantType::per_Token;
    void *inter_states_ptr = inter_states.data_ptr();
    void *w1_ptr = w1.data_ptr();
    void *w2_ptr = w2.data_ptr();
    void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
    void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
    void *sorted_weights_ptr = sorted_weights.data_ptr();
    void *num_valid_ids_ptr = num_valid_ids.data_ptr();
    void *out_ptr = out.data_ptr();
    void *w2_scale_ptr = w2_scale.has_value() ? w2_scale.value().data_ptr() : nullptr;
    void *a2_scale_ptr = a2_scale.has_value() ? a2_scale.value().data_ptr() : nullptr;
    if (!inter_states_ptr || !w1_ptr || !w2_ptr || !sorted_token_ids_ptr || !sorted_expert_ids_ptr || !num_valid_ids_ptr || !out_ptr)
    {
        std::cerr << "detect null ptr ！" << std::endl;
        return;
    }
    
    static const auto lookup = []
    {
        return MoeKernelMap{GENERATE_LOOKUP_TABLE()};
    }();

    auto it = lookup.find(kernelName);

    // If we found an optimal kernel, use it.
    if (it != lookup.end())
    {
        std::cout << "[aiter] found CK kernel : " << kernelName << std::endl;
        return it->second;
    }
    //std::cerr << "[aiter] ck kernel not found: " << kernelName << std::endl;
    // Otherwise, use heuristics.
    // BF16
    if (inter_states.dtype() == at::ScalarType::BFloat16)
    {
        if(q_type == QuantType::per_Tensor)
        {
            if (sorted_weights.has_value())
            {
                if(ACTOP == 0)
                {
                    if (MPerBlock == 32) 
                    {
                        using A0DataType = B16;
                        using B0DataType = B16;
                        using AccDataType = F32;
                        using EDataType = B16;
                        const bool Nswizzle = false;
                        using CDEElementOp = TypeCastExpertWeight;
                        if (K % (256 / sizeof(A0DataType)) == 0)
                        {
                            ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 256, 32, 128, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>
                            (at::cuda::getCurrentCUDAStream().stream(), 
                            tokens, sorted_size, N, K, topk, 
                            inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);

                        }
                    }
                }
            }
        }
    }

}