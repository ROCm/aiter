// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages_common.cuh"
#include "moe_cktile2stages_lookup.h"
#include "moe_cktile2stages_manifest_common.h"
#include "py_itfs_common.h"
#include "moe_cktile2stages_heuristic_dispatch_common.h"
#include <cmath>

#define MOE_DISPATCH_HEURISTIC(ADataType, BDataType, AccDataType, CDataType, stage, act, bias, split, bnt) \
    if constexpr (stage == 1) { \
        return moe_gemm1_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, act, bias, split, bnt>::dispatch(M, N, K, block_m); \
    } else { \
        return moe_gemm2_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, act, bias, split, bnt>::dispatch(M, N, K, block_m); \
    }

#define MOE_DISPATCH_BNT_TYPE(ADataType, BDataType, AccDataType, CDataType, stage, act, bias, split, b_nt_type) \
    switch (b_nt_type) { \
        case 1: \
            MOE_DISPATCH_HEURISTIC(ADataType, BDataType, AccDataType, CDataType, stage, act, bias, split, 1); \
            break; \
        case 2: \
            MOE_DISPATCH_HEURISTIC(ADataType, BDataType, AccDataType, CDataType, stage, act, bias, split, 2); \
            break; \
        default: \
            MOE_DISPATCH_HEURISTIC(ADataType, BDataType, AccDataType, CDataType, stage, act, bias, split, 0); \
            break; \
    }

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          int stage = 1>
MoeKernel moe_dispatch(int M, int N, int K, int block_m, int activation, bool has_bias, int split_k, int b_nt_type)
{
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.

    // Otherwise, use heuristics with macro-based dispatch.
    if (split_k > 1)
    {
        if (activation == 2 && has_bias) 
        {
            MOE_DISPATCH_BNT_TYPE(ADataType, BDataType, AccDataType, CDataType, stage, 2, true, true, b_nt_type);
        }
        else if (activation == 2 && !has_bias) 
        {
            MOE_DISPATCH_BNT_TYPE(ADataType, BDataType, AccDataType, CDataType, stage, 2, false, true, b_nt_type);
        }
        else if (activation == 0 && has_bias) 
        {
            MOE_DISPATCH_BNT_TYPE(ADataType, BDataType, AccDataType, CDataType, stage, 0, true, true, b_nt_type);
        }
        else if (activation == 0 && !has_bias) 
        {
            MOE_DISPATCH_BNT_TYPE(ADataType, BDataType, AccDataType, CDataType, stage, 0, false, true, b_nt_type);
        }
    }
    else
    {
        if (activation == 2 && has_bias) 
        {
            MOE_DISPATCH_BNT_TYPE(ADataType, BDataType, AccDataType, CDataType, stage, 2, true, false, b_nt_type);
        }
        else if (activation == 2 && !has_bias) 
        {
            MOE_DISPATCH_BNT_TYPE(ADataType, BDataType, AccDataType, CDataType, stage, 2, false, false, b_nt_type);
        }
        else if (activation == 0 && has_bias) 
        {
            MOE_DISPATCH_BNT_TYPE(ADataType, BDataType, AccDataType, CDataType, stage, 0, true, false, b_nt_type);
        }
        else if (activation == 0 && !has_bias) 
        {
            MOE_DISPATCH_BNT_TYPE(ADataType, BDataType, AccDataType, CDataType, stage, 0, false, false, b_nt_type);
        }
    }
    
    // Should never reach here if all cases are covered
    TORCH_CHECK(false, "Unsupported combination of activation and has_bias");
}
    

torch::Tensor cktile_moe_gemm1(torch::Tensor& XQ,
                               torch::Tensor& WQ,
                               torch::Tensor& Y,
                               torch::Tensor& sorted_ids,
                               torch::Tensor& sorted_expert_ids,
                               torch::Tensor& max_token_ids,
                               int topk,
                               std::optional<int> n_padded_zeros,
                               std::optional<int> k_padded_zeros,
                               std::optional<torch::Tensor> topk_weight,
                               std::optional<torch::Tensor> x_scale,
                               std::optional<torch::Tensor> w_scale,
                               std::optional<torch::Tensor> exp_bias,
                               std::optional<int> activation,
                               std::optional<int> block_m,
                               std::optional<int> split_k,
                               std::optional<int> b_nt_type)
{
    TORCH_CHECK(Y.dtype() == at::ScalarType::BFloat16 || Y.dtype() == at::ScalarType::Half,
                "Out dtype only support BFloat16/Float16!");
    if(x_scale.has_value() && w_scale.has_value())
    {
        TORCH_CHECK(x_scale.value().dtype() == w_scale.value().dtype(),
                    "Scales should have the same dtype!");
    }
    int64_t token     = XQ.size(0);
    int M         = std::min(sorted_ids.size(0), token * topk * block_m.value());
    int N         = WQ.size(1);
    int K         = XQ.size(-1);
    int MPerBlock = block_m.has_value() ? block_m.value() : 32;

    bool has_bias = exp_bias.has_value();
    int act_op    = activation.has_value() ? activation.value() : -1;
    int k_batch   = split_k.has_value() ? split_k.value() : 1;
    int b_nt      = b_nt_type.has_value() ? b_nt_type.value() : 0;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(Y));
    at::hip::getCurrentHIPStream();
    // if (!XQ || !WQ || !sorted_ids || !sorted_expert_ids || !max_token_ids || !topk_weight)
    // {
    //     std::cerr << "detect null ptr !" << std::endl;
    //     return;
    // }

    if(XQ.dtype() == torch_fp8)
    {
        //     if (Y.dtype() == at::ScalarType::Half)
        //     {
        //        moe_dispatch<fp8, fp8, float, fp16, 1>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids,
        //        sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale, exp_bias);
        //     }
        // if (Y.dtype() == at::ScalarType::BFloat16)
        // {
        //     moe_dispatch<fp8, fp8, float, bf16, 1>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids,
        //     sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale, exp_bias);
        // }
        if (WQ.dtype() == torch_fp4x2 && Y.dtype() == at::ScalarType::BFloat16)
        {
            moe_dispatch<fp8, pk_fp4, float, bf16, 1>(M, N, K, MPerBlock, act_op, has_bias, k_batch, b_nt)(
                XQ,
                WQ,
                Y,
                sorted_ids,
                sorted_expert_ids,
                max_token_ids,
                topk,
                n_padded_zeros,
                k_padded_zeros,
                topk_weight,
                x_scale,
                w_scale,
                exp_bias,
                act_op,
                k_batch);
        }
    }
    else if((XQ.dtype() == at::ScalarType::BFloat16 || XQ.dtype() == at::ScalarType::Half) &&
            (WQ.dtype() == torch_fp4x2)) // a16w4
    {
        // if (Y.dtype() == at::ScalarType::Half)
        // {
        //    moe_dispatch<fp16, pk_fp4, float, fp16, 1>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids,
        //    sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale, exp_bias);
        // }
        if(Y.dtype() == at::ScalarType::BFloat16)
        {
            moe_dispatch<bf16, pk_fp4, float, bf16, 1>(M, N, K, MPerBlock, act_op, has_bias, k_batch, b_nt)(
                XQ,
                WQ,
                Y,
                sorted_ids,
                sorted_expert_ids,
                max_token_ids,
                topk,
                n_padded_zeros,
                k_padded_zeros,
                topk_weight,
                x_scale,
                w_scale,
                exp_bias,
                act_op,
                k_batch);
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}

torch::Tensor cktile_moe_gemm2(torch::Tensor& XQ,
                               torch::Tensor& WQ,
                               torch::Tensor& Y,
                               torch::Tensor& sorted_ids,
                               torch::Tensor& sorted_expert_ids,
                               torch::Tensor& max_token_ids,
                               int topk,
                               std::optional<int> n_padded_zeros,
                               std::optional<int> k_padded_zeros,
                               std::optional<torch::Tensor> topk_weight,
                               std::optional<torch::Tensor> x_scale,
                               std::optional<torch::Tensor> w_scale,
                               std::optional<torch::Tensor> exp_bias,
                               std::optional<int> activation,
                               std::optional<int> block_m,
                               std::optional<int> split_k,
                               std::optional<int> b_nt_type)
{
    int64_t token     = XQ.size(0);
    int MPerBlock = block_m.has_value() ? block_m.value() : 32;
    int M         = std::min(sorted_ids.size(0), token * topk * MPerBlock);
    int N         = WQ.size(1);
    int K         = XQ.size(-1);
    
    bool has_bias = exp_bias.has_value();
    int act_op    = activation.has_value() ? activation.value() : -1;
    int k_batch   = split_k.has_value() ? split_k.value() : 1;
    int b_nt      = b_nt_type.has_value() ? b_nt_type.value() : 0;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(Y));
    at::hip::getCurrentHIPStream();
    // if (!XQ. || !WQ || !sorted_ids || !sorted_expert_ids || !max_token_ids || !topk_weight)
    // {
    //     std::cerr << "detect null ptr !" << std::endl;
    //     return;
    // }

    if(XQ.dtype() == torch_fp8)
    {
        //     if (Y.dtype() == at::ScalarType::Half)
        //     {
        //        moe_dispatch<fp8, fp8, float, fp16, 2>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids,
        //        sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale, exp_bias);
        //     }
        // if (Y.dtype() == at::ScalarType::BFloat16)
        // {
        //     moe_dispatch<fp8, fp8, float, bf16, 2>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids,
        //     sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale, exp_bias);
        // }
        if (WQ.dtype() == torch_fp4x2 && Y.dtype() == at::ScalarType::BFloat16)
        {
            moe_dispatch<fp8, pk_fp4, float, bf16, 2>(M, N, K, MPerBlock, act_op, has_bias, k_batch, b_nt)(
                XQ,
                WQ,
                Y,
                sorted_ids,
                sorted_expert_ids,
                max_token_ids,
                topk,
                n_padded_zeros,
                k_padded_zeros,
                topk_weight,
                x_scale,
                w_scale,
                exp_bias,
                act_op,
                k_batch);
        }
    }
    else if((XQ.dtype() == at::ScalarType::BFloat16 || XQ.dtype() == at::ScalarType::Half) &&
            (WQ.dtype() == torch_fp4x2)) // a16w4
    {
        // if (Y.dtype() == at::ScalarType::Half)
        // {
        //    moe_dispatch<fp16, pk_fp4, float, fp16, 2>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids,
        //    sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale, exp_bias);
        // }
        if(Y.dtype() == at::ScalarType::BFloat16)
        {
            moe_dispatch<bf16, pk_fp4, float, bf16, 2>(M, N, K, MPerBlock, 0, has_bias, k_batch, b_nt)(
                XQ,
                WQ,
                Y,
                sorted_ids,
                sorted_expert_ids,
                max_token_ids,
                topk,
                n_padded_zeros,
                k_padded_zeros,
                topk_weight,
                x_scale,
                w_scale,
                exp_bias,
                act_op,
                k_batch);
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}
