// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "py_itfs_common.h"
#include "moe_ck_gemm.hpp"

#define CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock)                                                                                                                                                                                                                                        \
    if (MulRoutedWeight)                                                                                                                                                                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                                                                                                                                                       \
        if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
        }                                                                                                                                                                                                                                                                                                                                                                   \
        else                                                                                                                                                                                                                                                                                                                                                                \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
        }                                                                                                                                                                                                                                                                                                                                                                   \
    }                                                                                                                                                                                                                                                                                                                                                                       \
    else                                                                                                                                                                                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                                                                                                                                                       \
        if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
        }                                                                                                                                                                                                                                                                                                                                                                   \
        else                                                                                                                                                                                                                                                                                                                                                                \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
        }                                                                                                                                                                                                                                                                                                                                                                   \
    }
   
#define CK_MOE_STAGE1_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock)                                                                                                                                                                                                                                   \
    if (MulRoutedWeight)                                                                                                                                                                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                                                                                                                                                       \
        if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
        }                                                                                                                                                                                                                                                                                                                                                                   \
        else                                                                                                                                                                                                                                                                                                                                                                \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
        }                                                                                                                                                                                                                                                                                                                                                                   \
    }                                                                                                                                                                                                                                                                                                                                                                       \
    else                                                                                                                                                                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                                                                                                                                                       \
        if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
        }                                                                                                                                                                                                                                                                                                                                                                   \
        else                                                                                                                                                                                                                                                                                                                                                                \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
        }                                                                                                                                                                                                                                                                                                                                                                   \
    }                                                                                                                                                                                                                                                                                                                                                                       \

void ck_moe_stage1(torch::Tensor &hidden_states,     // [m, k], input token
                   torch::Tensor &w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                   torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                   torch::Tensor &num_valid_ids,     // [1]
                   torch::Tensor &out,               // [m * topk, inter_dim]
                   int topk,
                   std::optional<torch::Tensor> w1_scale = std::nullopt, // [e, 1, n], gate(up) scale
                   std::optional<torch::Tensor> a1_scale = std::nullopt, // [m, 1], token scale
                   std::optional<int> block_m = 32,
                   std::optional<torch::Tensor> sorted_weights = std::nullopt)    // [max_num_tokens_padded])
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
    bool isPerTensorQuant = (!w1_scale.has_value()) || (w1_scale.value().numel() == E);
    bool MulRoutedWeight = sorted_weights.has_value();

    // int M = agvtokens_per_expert < 32 ? 32 : (agvtokens_per_expert < 64 ? 64 : 128);

    void *hidden_states_ptr = hidden_states.data_ptr();
    void *w1_ptr = w1.transpose(1, 2).data_ptr();
    void *w2_ptr = w2.data_ptr();
    void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
    void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
    void *sorted_weights_ptr = MulRoutedWeight ? sorted_weights.value().data_ptr() : nullptr;
    void *num_valid_ids_ptr = num_valid_ids.data_ptr();
    void *out_ptr = out.data_ptr();
    void *w1_scale_ptr = w1_scale.has_value() ? w1_scale.value().data_ptr() : nullptr;
    void *a1_scale_ptr = a1_scale.has_value() ? a1_scale.value().data_ptr() : nullptr;

    // BF16
    if (hidden_states.dtype() == at::ScalarType::BFloat16)
    {
        using A0DataType = B16;
        using B0DataType = B16;
        using AccDataType = F32;
        using EDataType = B16;
        const bool Nswizzle = false;
        if (MulRoutedWeight) 
        {
            using CDEElementOp = TypeCastExpertWeight;
            CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else 
        {
            using CDEElementOp = TypeCast;
            CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
    // FP16
    else if (hidden_states.dtype() == at::ScalarType::Half)
    {
        using A0DataType = F16;
        using B0DataType = F16;
        using AccDataType = F32;
        using EDataType = F16;
        const bool Nswizzle = false;
        if (MulRoutedWeight) 
        {
            using CDEElementOp = TypeCastExpertWeight;
            CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else 
        {
            using CDEElementOp = TypeCast;
            CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
    // FP8 Wint4
    else if (hidden_states.dtype() == at::ScalarType::Float8_e4m3fnuz && w1.dtype() == at::ScalarType::UInt32)
    {
        using A0DataType = F8;
        using B0DataType = I4;
        const bool Nswizzle = false;
        TORCH_CHECK(a1_scale.has_value() && w1_scale.has_value(),
                    "MoE Quant must input scale!");
        TORCH_CHECK(a1_scale.value().dtype() == at::ScalarType::Float,
                    "Scales must be Float dtype!");
        using AccDataType = F32;
        using CDEElementOp = MulABScaleWint4;
        if (out.dtype() == at::ScalarType::Half)
        {
            CK_MOE_STAGE1_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else if (out.dtype() == at::ScalarType::BFloat16)
        {
            CK_MOE_STAGE1_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
    // FP8
    else if (hidden_states.dtype() == at::ScalarType::Float8_e4m3fnuz)
    {
        using A0DataType = F8;
        using B0DataType = F8;
        TORCH_CHECK(a1_scale.has_value() && w1_scale.has_value(),
                    "MoE Quant must input scale!");
        TORCH_CHECK(a1_scale.value().dtype() == at::ScalarType::Float,
                    "Scales must be Float dtype!");
        using AccDataType = F32;
        using CDEElementOp = MulABScale;
        const bool Nswizzle = false;
        if (out.dtype() == at::ScalarType::Half)
        {
            CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else if (out.dtype() == at::ScalarType::BFloat16)
        {
            CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
    // I8
    else if (hidden_states.dtype() == at::ScalarType::Char)
    {
        using A0DataType = I8;
        using B0DataType = I8;
        TORCH_CHECK(a1_scale.has_value() && w1_scale.has_value(),
                    "MoE Quant must input scale!");
        TORCH_CHECK(a1_scale.value().dtype() == at::ScalarType::Float,
                    "Scales must be Float dtype!");
        using AccDataType = I32;
        using CDEElementOp = MulABScale;
        const bool Nswizzle = false;
        if (out.dtype() == at::ScalarType::Half)
        {
            CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else if (out.dtype() == at::ScalarType::BFloat16)
        {
            CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
}

#define CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock)                                                                                                                                                                                                                                                       \
    if (MulRoutedWeight)                                                                                                                                                                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                                                                                                                                                       \
        if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
        }                                                                                                                                                                                                                                                                                                                                                                   \
        else                                                                                                                                                                                                                                                                                                                                                                \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
        }                                                                                                                                                                                                                                                                                                                                                                   \
    }                                                                                                                                                                                                                                                                                                                                                                       \
    else                                                                                                                                                                                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                                                                                                                                                       \
        if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
        }                                                                                                                                                                                                                                                                                                                                                                   \
        else                                                                                                                                                                                                                                                                                                                                                                \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
        }                                                                                                                                                                                                                                                                                                                                                                   \
    }

#define CK_MOE_STAGE2_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock)                                                                                                                                                                                                                                                  \
    if (MulRoutedWeight)                                                                                                                                                                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                                                                                                                                                       \
        if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
        }                                                                                                                                                                                                                                                                                                                                                                   \
        else                                                                                                                                                                                                                                                                                                                                                                \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
        }                                                                                                                                                                                                                                                                                                                                                                   \
    }                                                                                                                                                                                                                                                                                                                                                                       \
    else                                                                                                                                                                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                                                                                                                                                       \
        if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
        }                                                                                                                                                                                                                                                                                                                                                                   \
        else                                                                                                                                                                                                                                                                                                                                                                \
        {                                                                                                                                                                                                                                                                                                                                                                   \
            if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
            else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
            else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
                ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
        }                                                                                                                                                                                                                                                                                                                                                                   \
    }  

void ck_moe_stage2(torch::Tensor &inter_states,      // [m, k], input token
                   torch::Tensor &w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                   torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                   torch::Tensor &num_valid_ids,     // [1]
                   torch::Tensor &out,               // [max_num_tokens_padded, inter_dim]
                   int topk,
                   std::optional<torch::Tensor> w2_scale = std::nullopt, // [e, 1, n], gate(up) scale
                   std::optional<torch::Tensor> a2_scale = std::nullopt, // [m, 1], token scale
                   std::optional<int> block_m = 32,
                   std::optional<torch::Tensor> sorted_weights = std::nullopt)    // [max_num_tokens_padded])
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
    bool MulRoutedWeight = sorted_weights.has_value();
    

    void *inter_states_ptr = inter_states.data_ptr();
    void *w1_ptr = w1.data_ptr();
    void *w2_ptr = w2.data_ptr();
    void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
    void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
    void *sorted_weights_ptr = MulRoutedWeight ? sorted_weights.value().data_ptr() : nullptr;
    void *num_valid_ids_ptr = num_valid_ids.data_ptr();
    void *out_ptr = out.data_ptr();
    void *w2_scale_ptr = w2_scale.has_value() ? w2_scale.value().data_ptr() : nullptr;
    void *a2_scale_ptr = a2_scale.has_value() ? a2_scale.value().data_ptr() : nullptr;

    // BF16
    if (inter_states.dtype() == at::ScalarType::BFloat16)
    {
        using A0DataType = B16;
        using B0DataType = B16;
        using AccDataType = F32;
        using EDataType = B16;
        const bool Nswizzle = false;
        if (MulRoutedWeight) 
        {
            using CDEElementOp = TypeCastExpertWeight;
            CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else 
        {
            using CDEElementOp = TypeCast;
            CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
    // FP16
    else if (inter_states.dtype() == at::ScalarType::Half)
    {
        using A0DataType = F16;
        using B0DataType = F16;
        using AccDataType = F32;
        using EDataType = F16;
        const bool Nswizzle = false;
        if (MulRoutedWeight) 
        {
            using CDEElementOp = TypeCastExpertWeight;
            CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else 
        {
            using CDEElementOp = TypeCast;
            CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
    // FP8 wint4
    else if (inter_states.dtype() == at::ScalarType::Float8_e4m3fnuz && w1.dtype() == at::ScalarType::UInt32)
    {
        using A0DataType = F8;
        using B0DataType = I4;
        const bool Nswizzle = false;
        TORCH_CHECK(a2_scale.has_value() && w2_scale.has_value(),
                    "MoE Quant must input scale!");
        TORCH_CHECK(a2_scale.value().dtype() == at::ScalarType::Float,
                    "Scales must be Float dtype!");
        using AccDataType = F32;
        using CDEElementOp = MulABScaleExpertWeightWin4;
        if (out.dtype() == at::ScalarType::Half)
        {
            CK_MOE_STAGE2_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else if (out.dtype() == at::ScalarType::BFloat16)
        {
            CK_MOE_STAGE2_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
    // FP8
    else if (inter_states.dtype() == at::ScalarType::Float8_e4m3fnuz)
    {
        using A0DataType = F8;
        using B0DataType = F8;
        TORCH_CHECK(a2_scale.has_value() && w2_scale.has_value(),
                    "MoE Quant must input scale!");
        TORCH_CHECK(a2_scale.value().dtype() == at::ScalarType::Float,
                    "Scales must be Float dtype!");
        using AccDataType = F32;
        using CDEElementOp = MulABScaleExpertWeight;
        const bool Nswizzle = false;
        if (out.dtype() == at::ScalarType::Half)
        {
            CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else if (out.dtype() == at::ScalarType::BFloat16)
        {
            CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
    // I8
    else if (inter_states.dtype() == at::ScalarType::Char)
    {
        using A0DataType = I8;
        using B0DataType = I8;
        TORCH_CHECK(a2_scale.has_value() && w2_scale.has_value(),
                    "MoE Quant must input scale!");
        TORCH_CHECK(a2_scale.value().dtype() == at::ScalarType::Float,
                    "Scales must be Float dtype!");
        using AccDataType = I32;
        using CDEElementOp = MulABScaleExpertWeight;
        const bool Nswizzle = false;
        if (out.dtype() == at::ScalarType::Half)
        {
            CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
        else if (out.dtype() == at::ScalarType::BFloat16)
        {
            CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, Nswizzle, isPerTensorQuant, MPerBlock);
        }
    }
}
