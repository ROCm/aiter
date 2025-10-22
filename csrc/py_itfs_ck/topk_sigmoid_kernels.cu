// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "py_itfs_common.h"

// from CK examples:
#include "topk_softmax_api.hpp"

void topk_sigmoid(torch::Tensor topk_weights,   // [tokens, topk]
                  torch::Tensor topk_indices,   // [tokens, topk]
                  torch::Tensor gating_output)  // [tokens, experts] 
{
    // Ensure the tensors are on the correct device
    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract dimensions
    const int tokens  = gating_output.size(0);
    const int experts = gating_output.size(1);
    const int topk    = topk_weights.size(1);
    
    // Assume default strides
    const int stride_input  = experts;
    const int stride_output = topk;

    // Prepare kernel arguments
    topk_softmax_trait trait{input_prec, weight_prec, experts};

    topk_softmax_kargs karg{gating_output.data_ptr(),
                            topk_weights.data_ptr(),
                            topk_indices.data_ptr(),
                            tokens,
                            experts,
                            topk,
                            stride_input,
                            stride_output};

    ck_tile::stream_config sc{stream};
  
    topk_softmax(trait, karg, sc);
}
