#pragma once

torch::Tensor all_reduce_asm(torch::Tensor &input,
                             int64_t _ca,
                             torch::Tensor &reg_sig, torch::Tensor &reg_buffer, bool isGraph);

std::tuple<torch::Tensor, torch::Tensor> all_reduce_rmsnorm(torch::Tensor &input,       // [m ,n]
                                                              torch::Tensor &residual_in, // [m ,n]
                                                              torch::Tensor &weight,      // [1 ,n]
                                                              torch::Tensor &bias,        // [1 ,n]
                                                              float epsilon,
                                                              // following are fused_allreduce args
                                                              int64_t _ca,
                                                              torch::Tensor &reg_sig, torch::Tensor &reg_buffer, bool isGraph);