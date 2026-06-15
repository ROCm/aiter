// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>
#include "custom_all_reduce.h"

namespace aiter {

// Native AllReduce + MHC(post+pre) + RMSNorm entry point.
void fused_allreduce_mhc_fused_post_pre_rmsnorm(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& layer_input,
    torch::Tensor& residual_in,
    torch::Tensor& post_layer_mix,
    torch::Tensor& comb_res_mix,
    torch::Tensor& fn,
    torch::Tensor& hc_scale,
    torch::Tensor& hc_base,
    torch::Tensor& norm_weight,
    torch::Tensor& gemm_out,
    torch::Tensor& gemm_out_sqrsum,
    torch::Tensor& next_residual,
    torch::Tensor& post_mix,
    torch::Tensor& comb_mix,
    torch::Tensor& layer_input_out,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float norm_eps,
    float hc_post_mult_value,
    int sinkhorn_repeat,
    int tile_m,
    int tile_n,
    int tile_k,
    int pre_tile_k,
    int post_store_nt,
    bool use_large_m,
    bool use_ar_mhc_full_fusion,
    bool use_ar_mhc_post_epilogue,
    bool use_large_m_post_epilogue,
    bool use_new,
    bool open_fp8_quant,
    int64_t reg_ptr,
    int64_t reg_bytes);

} // namespace aiter
