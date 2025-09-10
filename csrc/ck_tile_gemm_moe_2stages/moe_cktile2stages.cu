// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages_common.cuh"
#include "moe_cktile2stages_lookup.h"
#include "moe_cktile2stages_manifest.h"
#include "moe_cktile2stages_heuristic_dispatch.h"
#include <cmath>
#include "py_itfs_common.h"

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType, int stage = 1>
MoeKernel moe_dispatch(int M, int N, int K, int block_m)
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
  if (stage == 1){
    return moe_gemm1_heuristic_dispatch<ADataType, BDataType, AccDataType, CDataType>(M, N, K, block_m);
  }
  else{
    return moe_gemm2_heuristic_dispatch<ADataType, BDataType, AccDataType, CDataType>(M, N, K, block_m);
  }
}



torch::Tensor cktile_moe_gemm1(torch::Tensor& XQ,
                        torch::Tensor& WQ,
                        torch::Tensor& Y,
                        torch::Tensor& sorted_ids,
                        torch::Tensor& sorted_expert_ids,
                        torch::Tensor& max_token_ids,
                        int topk,
                        std::optional<torch::Tensor> topk_weight,
                        std::optional<torch::Tensor> x_scale,
                        std::optional<torch::Tensor> w_scale,
                        std::optional<int> block_m)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(),
                "Weights and activations should both be the same.");
            
    TORCH_CHECK(Y.dtype() == at::ScalarType::BFloat16 || Y.dtype() == at::ScalarType::Half,
                "Out dtype only support BFloat16/Float16!");
    if (x_scale != std::nullopt && w_scale != std::nullopt){
        TORCH_CHECK(x_scale.value().dtype() == w_scale.value().dtype(),
                    "Scales should have the same dtype!");
    }
    int M = reinterpret_cast<const ck_tile::index_t*>(max_token_ids.data_ptr())[0];
    int N = WQ.size(1);
    int K = XQ.size(-1);
    int MPerBlock = block_m.value();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Y));
    at::cuda::getCurrentCUDAStream().stream();
    // if (!XQ || !WQ || !sorted_ids || !sorted_expert_ids || !max_token_ids || !topk_weight)
    // {
    //     std::cerr << "detect null ptr !" << std::endl;
    //     return;
    // }

    if (XQ.dtype() == torch_fp8)
    {
        if (Y.dtype() == at::ScalarType::Half)
        {
           moe_dispatch<fp8, fp8, float, fp16, 1>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids, sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale); 
        }
        else if (Y.dtype() == at::ScalarType::BFloat16)
        {
            moe_dispatch<fp8, fp8, float, bf16, 1>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids, sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale); 
        }
    }
    else if ((XQ.dtype() == at::ScalarType::BFloat16 || XQ.dtype() == at::ScalarType::Half) && (WQ.dtype() == at::ScalarType::Byte)) //a16w4
    {
        if (Y.dtype() == at::ScalarType::Half)
        {
           moe_dispatch<fp16, pk_fp4, float, fp16, 1>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids, sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale); 
        }
        else if (Y.dtype() == at::ScalarType::BFloat16)
        {
            moe_dispatch<bf16, pk_fp4, float, bf16, 1>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids, sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale); 
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
                        std::optional<torch::Tensor> topk_weight,
                        std::optional<torch::Tensor> x_scale,
                        std::optional<torch::Tensor> w_scale,
                        std::optional<int> block_m)
{
    int M = XQ.size(0) * XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(-1);
    int MPerBlock = block_m.value();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Y));
    at::cuda::getCurrentCUDAStream().stream();
    // if (!XQ. || !WQ || !sorted_ids || !sorted_expert_ids || !max_token_ids || !topk_weight)
    // {
    //     std::cerr << "detect null ptr !" << std::endl;
    //     return;
    // }

    if (XQ.dtype() == torch_fp8)
    {
        if (Y.dtype() == at::ScalarType::Half)
        {
           moe_dispatch<fp8, fp8, float, fp16, 2>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids, sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale); 
        }
        else if (Y.dtype() == at::ScalarType::BFloat16)
        {
            moe_dispatch<fp8, fp8, float, bf16, 2>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids, sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale); 
        }
    }
    else if ((XQ.dtype() == at::ScalarType::BFloat16 || XQ.dtype() == at::ScalarType::Half) && (WQ.dtype() == at::ScalarType::Byte)) //a16w4
    {
        if (Y.dtype() == at::ScalarType::Half)
        {
           moe_dispatch<fp16, pk_fp4, float, fp16, 2>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids, sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale); 
        }
        else if (Y.dtype() == at::ScalarType::BFloat16)
        {
            moe_dispatch<bf16, pk_fp4, float, bf16, 2>(M, N, K, MPerBlock)(XQ, WQ, Y, sorted_ids, sorted_expert_ids, max_token_ids, topk, topk_weight, x_scale, w_scale); 
        }
    }
    else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}