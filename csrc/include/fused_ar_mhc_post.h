// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"
#include "custom_all_reduce.h"

namespace aiter {

void fused_allreduce_mhc_post_only(fptr_t _fa,
                                   aiter_tensor_t& inp,
                                   aiter_tensor_t& next_residual,
                                   aiter_tensor_t& residual_in,
                                   aiter_tensor_t& post_layer_mix,
                                   aiter_tensor_t& comb_res_mix,
                                   bool use_new,
                                   bool open_fp8_quant,
                                   int64_t reg_ptr,
                                   int64_t reg_bytes);

void fused_allreduce_mhc_post_one_stage(fptr_t _fa,
                                        aiter_tensor_t& inp,
                                        aiter_tensor_t& next_residual,
                                        aiter_tensor_t& residual_in,
                                        aiter_tensor_t& post_layer_mix,
                                        aiter_tensor_t& comb_res_mix,
                                        bool use_new,
                                        bool open_fp8_quant,
                                        int64_t reg_ptr,
                                        int64_t reg_bytes);

void fused_allreduce_mhc_post_split(fptr_t _fa,
                                    aiter_tensor_t& inp,
                                    aiter_tensor_t& next_residual,
                                    aiter_tensor_t& residual_in,
                                    aiter_tensor_t& post_layer_mix,
                                    aiter_tensor_t& comb_res_mix,
                                    bool use_new,
                                    bool open_fp8_quant,
                                    int64_t reg_ptr,
                                    int64_t reg_bytes);

} // namespace aiter
