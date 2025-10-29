// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include "py_itfs_common.h"

// from CK examples:
#include "topk_softmax_api.hpp"

namespace aiter
{

void topk_sigmoid(torch::Tensor topk_weights,   // [tokens, topk]
                  torch::Tensor topk_indices,   // [tokens, topk]
                  torch::Tensor gating_output)  // [tokens, experts] 
{
    // Ensure the tensors are on the correct device
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));

    // Extract dimensions
    const int tokens  = gating_output.size(0);
    const int experts = gating_output.size(1);
    const int topk    = topk_weights.size(1);
    
    // Assume default strides
    const int stride_input  = experts;
    const int stride_output = topk;

    // Determine datatypes
    auto dtype_to_string = [](const auto dtype) -> std::string {
        if(dtype == torch::kFloat16)
        {
            return "fp16";
        }
        else if(dtype == torch::kBFloat16)
        {
            return "bf16";
        }
        else if(dtype == torch::kFloat32)
        {
            return "fp32";
        }
        else
        {
            throw std::runtime_error("invalid datatype for topk_sigmoid: only fp16/bf16/fp32!");
        }
    };
    std::string input_prec  = dtype_to_string(gating_output.dtype());
    std::string weight_prec = dtype_to_string(topk_weights.dtype());

    // Prepare kernel arguments
    static const std::string activation = "sigmoid";
    topk_softmax_trait trait{input_prec, weight_prec, experts, activation};

    topk_softmax_kargs karg{gating_output.data_ptr(),
                            topk_weights.data_ptr(),
                            topk_indices.data_ptr(),
                            tokens,
                            experts,
                            topk,
                            stride_input,
                            stride_output};

    ck_tile::stream_config sc{at::hip::getCurrentHIPStream()};
  
    topk_softmax(trait, karg, sc);
}

} // namespace aiter
