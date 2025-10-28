#pragma once
/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2025, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/extension.h>

void rmsnorm2d_with_add_smoothquant_hip(
    torch::Tensor& out,          // [m ,n]
    torch::Tensor& input,        // [m ,n]
    torch::Tensor& residual_in,  // [m ,n]
    torch::Tensor& residual_out, // [m ,n]
    torch::Tensor& xscale,       // [1 ,n]
    torch::Tensor& yscale,       // [m ,1]
    torch::Tensor& weight,       // [1 ,n]
    double epsilon,
    std::optional<torch::Tensor> out_before_quant,
    int use_model_sensitive_rmsnorm = 0); // 0: Use default RMSNorm; 1: Use T5-like implementation
                                          //
torch::Tensor
rmsnorm2d_hip(torch::Tensor& input,
              torch::Tensor& weight,
              double epsilon,
              int use_model_sensitive_rmsnorm = 0); // 0: Use default RMSNorm; 1: Use T5-like implementation
