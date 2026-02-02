#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_cooperative_groups.h>
#include "aiter_hip_common.h"

namespace cooperative_groups {
template <typename T>
struct plus {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T, typename Op>
__device__ __forceinline__ T reduce(::cooperative_groups::thread_block_tile<32> tile, T val, Op op)
{
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset);
        val = op(val, other);
    }
    return val;
}
}  // namespace cooperative_groups

namespace aiter {

struct MHCConfig {
    int sinkhorn_iters;
    int nC;
    float eps;
    bool use_pdl;
};

struct RMSNormParams {
    int n;
    float eps;
};

void mhc_layer_fwd(torch::Tensor &out,
                   torch::Tensor &x_expanded,
                   torch::Tensor &rmsnorm_weight,
                   torch::Tensor &phi_pre,
                   torch::Tensor &phi_post,
                   torch::Tensor &phi_res,
                   torch::Tensor &b_pre,
                   torch::Tensor &b_post,
                   torch::Tensor &b_res,
                   double alpha_pre,
                   double alpha_post,
                   double alpha_res,
                   int64_t sinkhorn_iters,
                   double eps,
                   bool use_pdl);

void mhc_layer_fwd_debug(torch::Tensor &out,
                         torch::Tensor &x_expanded,
                         torch::Tensor &rmsnorm_weight,
                         torch::Tensor &phi_pre,
                         torch::Tensor &phi_post,
                         torch::Tensor &phi_res,
                         torch::Tensor &b_pre,
                         torch::Tensor &b_post,
                         torch::Tensor &b_res,
                         double alpha_pre,
                         double alpha_post,
                         double alpha_res,
                         int64_t sinkhorn_iters,
                         double eps,
                         torch::Tensor &H_proj_raw,
                         torch::Tensor &H_pre,
                         torch::Tensor &H_post,
                         torch::Tensor &M,
                         torch::Tensor &x_agg_bf16,
                         torch::Tensor &layer_out_bf16,
                         torch::Tensor &rms_values,
                         bool use_pdl);

} // namespace aiter
