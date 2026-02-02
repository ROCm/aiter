#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

namespace aiter {

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

} // namespace aiter
