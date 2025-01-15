#pragma once

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
                                    double epsilon);

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
