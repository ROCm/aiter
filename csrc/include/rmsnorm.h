#pragma once
/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2024, The vLLM team.
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

void rms_norm(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor &input, torch::Tensor &residual,
                        torch::Tensor &weight, double epsilon);

// ck
torch::Tensor rmsnorm2d(torch::Tensor &input, torch::Tensor &weight,
                        double epsilon);

void rmsnorm2d_with_add(torch::Tensor &out,          // [m ,n]
                        torch::Tensor &input,        // [m ,n]
                        torch::Tensor &residual_in,  // [m ,n]
                        torch::Tensor &residual_out, // [m ,n]
                        torch::Tensor &weight,       // [1 ,n]
                        double epsilon);

void rmsnorm2d_with_smoothquant(torch::Tensor &out,    // [m ,n]
                                torch::Tensor &input,  // [m ,n]
                                torch::Tensor &xscale, // [1 ,n]
                                torch::Tensor &yscale, // [m ,1]
                                torch::Tensor &weight, // [1 ,n]
                                double epsilon);

void rmsnorm2d_with_add_smoothquant(torch::Tensor &out,          // [m ,n]
                                    torch::Tensor &input,        // [m ,n]
                                    torch::Tensor &residual_in,  // [m ,n]
                                    torch::Tensor &residual_out, // [m ,n]
                                    torch::Tensor &xscale,       // [1 ,n]
                                    torch::Tensor &yscale,       // [m ,1]
                                    torch::Tensor &weight,       // [1 ,n]
                                    double epsilon,
                                    std::optional<torch::Tensor> out_before_quant);

void rmsnorm2d_with_dynamicquant(torch::Tensor &out,    // [m ,n]
                                 torch::Tensor &input,  // [m ,n]
                                 torch::Tensor &yscale, // [m ,1]
                                 torch::Tensor &weight, // [1 ,n]
                                 double epsilon);

void rmsnorm2d_with_add_dynamicquant(torch::Tensor &out,          // [m ,n]
                                     torch::Tensor &input,        // [m ,n]
                                     torch::Tensor &residual_in,  // [m ,n]
                                     torch::Tensor &residual_out, // [m ,n]
                                     torch::Tensor &yscale,       // [m ,1]
                                     torch::Tensor &weight,       // [1 ,n]
                                     double epsilon);
