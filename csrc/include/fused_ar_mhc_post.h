// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>
#include "custom_all_reduce.h"

namespace aiter {

void fused_allreduce_mhc_post_only(fptr_t _fa,
                                   torch::Tensor& inp,
                                   torch::Tensor& next_residual,
                                   torch::Tensor& residual_in,
                                   torch::Tensor& post_layer_mix,
                                   torch::Tensor& comb_res_mix,
                                   bool use_new,
                                   bool open_fp8_quant,
                                   int64_t reg_ptr,
                                   int64_t reg_bytes);

void fused_allreduce_mhc_post_one_stage(fptr_t _fa,
                                        torch::Tensor& inp,
                                        torch::Tensor& next_residual,
                                        torch::Tensor& residual_in,
                                        torch::Tensor& post_layer_mix,
                                        torch::Tensor& comb_res_mix,
                                        bool use_new,
                                        bool open_fp8_quant,
                                        int64_t reg_ptr,
                                        int64_t reg_bytes);

void fused_allreduce_mhc_post_split(fptr_t _fa,
                                    torch::Tensor& inp,
                                    torch::Tensor& next_residual,
                                    torch::Tensor& residual_in,
                                    torch::Tensor& post_layer_mix,
                                    torch::Tensor& comb_res_mix,
                                    bool use_new,
                                    bool open_fp8_quant,
                                    int64_t reg_ptr,
                                    int64_t reg_bytes);

} // namespace aiter
