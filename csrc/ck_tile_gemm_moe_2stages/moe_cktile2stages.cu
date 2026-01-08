// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages_common.cuh"
#include "moe_cktile2stages_lookup.h"
#include "moe_cktile2stages_manifest_common.h"
#include "py_itfs_common.h"
#include "moe_cktile2stages_heuristic_dispatch_common.h"
#include <cmath>

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          int stage = 1>
MoeKernel moe_dispatch(int M, int N, int K, int block_m, int activation, bool has_bias, int split_k)
{
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.

    // static const auto lookup = [&]
    // {
    //   return RowwiseKernelMap{GENERATE_LOOKUP_TABLE(ABDataType, AccDataType, CDataType)};
    // }();

    // // First check if this shape(M,N,K) is available in the direct lookup.
    // auto it = lookup.find({M, N, K});
    // // If we found an optimal kernel, use it.
    // if (it != lookup.end())
    // {
    //   return it->second;
    // }

    // int padded_m = M;
    // if (M > 1 && M <= 16)
    // {
    //   padded_m = 16;
    // }
    // else if (M <= 16384)
    // {
    //   padded_m = nextPow2(M);
    // }
    // else if (M <= 20480)
    // {
    //   padded_m = 20480;
    // }
    // // Second check if this shape(padded_m,N,K) is available in the direct lookup.
    // it = lookup.find({padded_m, N, K});
    // // If we found an optimal kernel, use it.
    // if (it != lookup.end())
    // {
    //   return it->second;
    // }
    // Otherwise, use heuristics.
    if (split_k > 1)
    {
        if (activation == 2 && has_bias) 
        {
            if (stage == 1)
            {
                return moe_gemm1_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 2, true, true>::dispatch(
                    M, N, K, block_m);
            }
            else
            {
                return moe_gemm2_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 2, true, true>::dispatch(
                    M, N, K, block_m);
            }
        }
        else if (activation == 2 && !has_bias) {
            if (stage == 1)
            {
                return moe_gemm1_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 2, false, true>::dispatch(
                    M, N, K, block_m);
            }
            else
            {
                return moe_gemm2_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 2, false, true>::dispatch(
                    M, N, K, block_m);
            }
        }
        else if (activation == 0 && has_bias) {
            if (stage == 1)
            {
                return moe_gemm1_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 0, true, true>::dispatch(
                    M, N, K, block_m);
            }
            else
            {
                return moe_gemm2_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 0, true, true>::dispatch(
                    M, N, K, block_m);
            }
        }
        else if (activation == 0 && !has_bias) {
            if (stage == 1)
            {
                return moe_gemm1_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 0, false, true>::dispatch(
                    M, N, K, block_m);
            }
            else
            {
                return moe_gemm2_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 0, false, true>::dispatch(
                    M, N, K, block_m);
            }
        }
    }
    else
    {
        if (activation == 2 && has_bias) 
        {
            if (stage == 1)
            {
                return moe_gemm1_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 2, true, false>::dispatch(
                    M, N, K, block_m);
            }
            else
            {
                return moe_gemm2_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 2, true, false>::dispatch(
                    M, N, K, block_m);
            }
        }
        else if (activation == 2 && !has_bias) {
            if (stage == 1)
            {
                return moe_gemm1_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 2, false, false>::dispatch(
                    M, N, K, block_m);
            }
            else
            {
                return moe_gemm2_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 2, false, false>::dispatch(
                    M, N, K, block_m);
            }
        }
        else if (activation == 0 && has_bias) {
            if (stage == 1)
            {
                return moe_gemm1_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 0, true, false>::dispatch(
                    M, N, K, block_m);
            }
            else
            {
                return moe_gemm2_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 0, true, false>::dispatch(
                    M, N, K, block_m);
            }
        }
        else if (activation == 0 && !has_bias) {
            if (stage == 1)
            {
                return moe_gemm1_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 0, false, false>::dispatch(
                    M, N, K, block_m);
            }
            else
            {
                return moe_gemm2_heuristic_dispatcher<ADataType, BDataType, AccDataType, CDataType, 0, false, false>::dispatch(
                    M, N, K, block_m);
            }
        }
    }
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
                               std::optional<int> split_k)
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
            moe_dispatch<fp8, pk_fp4, float, bf16, 1>(M, N, K, MPerBlock, act_op, has_bias, k_batch)(
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
            moe_dispatch<bf16, pk_fp4, float, bf16, 1>(M, N, K, MPerBlock, act_op, has_bias, k_batch)(
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
                               std::optional<int> split_k)
{
    int64_t token     = XQ.size(0);
    int MPerBlock = block_m.has_value() ? block_m.value() : 32;
    int M         = std::min(sorted_ids.size(0), token * topk * MPerBlock);
    int N         = WQ.size(1);
    int K         = XQ.size(-1);
    
    bool has_bias = exp_bias.has_value();
    int act_op    = activation.has_value() ? activation.value() : -1;
    int k_batch   = split_k.has_value() ? split_k.value() : 1;

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
            moe_dispatch<fp8, pk_fp4, float, bf16, 2>(M, N, K, MPerBlock, act_op, has_bias, k_batch)(
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
            moe_dispatch<bf16, pk_fp4, float, bf16, 2>(M, N, K, MPerBlock, 0, has_bias, k_batch)(
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
