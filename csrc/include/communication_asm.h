#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
torch::Tensor all_reduce_asm(torch::Tensor &input,
                             int64_t _ca,
                             torch::Tensor &reg_sig, torch::Tensor &reg_buffer, bool isGraph);

std::tuple<torch::Tensor, torch::Tensor>       // out, residual_out
all_reduce_rmsnorm(torch::Tensor &input,       // [m ,n]
                   torch::Tensor &residual_in, // [m ,n]
                   torch::Tensor &weight,      // [1 ,n]
                   torch::Tensor &bias,        // [1 ,n]
                   float epsilon,
                   // following are fused_allreduce args
                   int64_t _ca,
                   torch::Tensor &reg_sig, torch::Tensor &reg_buffer, bool isGraph);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> // out, residual_out, yscale
all_reduce_rmsnorm_quant(torch::Tensor &input,          // [m ,n]
                         torch::Tensor &residual_in,    // [m ,n]
                         torch::Tensor &xscale,         // [1 ,n]
                         torch::Tensor &weight,         // [1 ,n]
                         torch::Tensor &bias,           // [1 ,n]
                         float epsilon,
                         // following are fused_allreduce args
                         int64_t _ca,
                         torch::Tensor &reg_sig, torch::Tensor &reg_buffer, bool isGraph);