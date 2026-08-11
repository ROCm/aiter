// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_backward_common.cuh"
#include "aiter_tensor.h"

#include <hip/hip_runtime.h>

namespace opus_moe_backward
{

// Internal launch API.  The future tensor/Python binding is responsible for
// validating tensor dtype/shape/device and constructing these plain kargs.
// Keeping this boundary small lets each family land and be benchmarked alone.
void launch_down_bwd_bf16(const DownBwdKargs& kargs,
                          int kernel_id,
                          hipStream_t stream);
void launch_route_dx_bf16(const RouteDxKargs& kargs,
                          int kernel_id,
                          hipStream_t stream);
void launch_route_reduce_bf16(const RouteReduceKargs& kargs,
                              int kernel_id,
                              hipStream_t stream);
void launch_dw1_bf16(const Dw1Kargs& kargs, int kernel_id, hipStream_t stream);
void launch_dw2_bf16(const Dw2Kargs& kargs, int kernel_id, hipStream_t stream);
void launch_router_bwd_fp32(const RouterBwdKargs& kargs,
                            int kernel_id,
                            hipStream_t stream);
void launch_bias_bwd_bf16(const BiasBwdKargs& kargs,
                          int kernel_id,
                          hipStream_t stream);

} // namespace opus_moe_backward

// Selected-softmax Jacobian plus scatter into the full router-logit gradient.
// topk_ids must be the exact fixed routes selected by forward.
void opus_moe_router_bwd(aiter_tensor_t& d_scores,
                         aiter_tensor_t& scores,
                         aiter_tensor_t& topk_ids,
                         aiter_tensor_t& d_logits,
                         int kernel_id);

void opus_moe_db1_bwd(aiter_tensor_t& d_z,
                      aiter_tensor_t& sorted_token_ids,
                      aiter_tensor_t& num_valid_ids,
                      aiter_tensor_t& expert_padded_offsets,
                      aiter_tensor_t& d_b1,
                      int token_num,
                      int topk,
                      int block_m,
                      int kernel_id);

void opus_moe_bias_down_bwd(aiter_tensor_t& d_out,
                            aiter_tensor_t& scores,
                            aiter_tensor_t& b2,
                            aiter_tensor_t& sorted_token_ids,
                            aiter_tensor_t& sorted_expert_ids,
                            aiter_tensor_t& num_valid_ids,
                            aiter_tensor_t& expert_padded_offsets,
                            aiter_tensor_t& d_scores,
                            aiter_tensor_t& d_b2,
                            int block_m,
                            int kernel_id);

void opus_moe_varlen_down_bwd(aiter_tensor_t& d_out,
                              aiter_tensor_t& z,
                              aiter_tensor_t& w2,
                              aiter_tensor_t& scores,
                              aiter_tensor_t& b2,
                              aiter_tensor_t& sorted_route_ids,
                              aiter_tensor_t& sorted_expert_ids,
                              aiter_tensor_t& num_valid_ids,
                              aiter_tensor_t& route_to_token,
                              aiter_tensor_t& expert_padded_offsets,
                              aiter_tensor_t& d_scores_workspace,
                              aiter_tensor_t& d_z,
                              aiter_tensor_t& a_scaled,
                              aiter_tensor_t& d_scores,
                              aiter_tensor_t& d_w2,
                              aiter_tensor_t& d_b2,
                              int block_m,
                              bool has_bias,
                              bool compute_dz,
                              bool compute_dw2,
                              bool compute_dscore,
                              bool compute_db2,
                              int down_kernel_id,
                              int dw2_kernel_id,
                              int bias_kernel_id);

void opus_moe_varlen_up_bwd(aiter_tensor_t& d_z,
                            aiter_tensor_t& x,
                            aiter_tensor_t& w1,
                            aiter_tensor_t& sorted_route_ids,
                            aiter_tensor_t& sorted_expert_ids,
                            aiter_tensor_t& num_valid_ids,
                            aiter_tensor_t& route_to_token,
                            aiter_tensor_t& token_route_offsets,
                            aiter_tensor_t& expert_padded_offsets,
                            aiter_tensor_t& d_x_route,
                            aiter_tensor_t& d_x,
                            aiter_tensor_t& d_w1,
                            aiter_tensor_t& d_b1,
                            int block_m,
                            bool compute_dx,
                            bool compute_dw1,
                            bool compute_db1,
                            int route_dx_kernel_id,
                            int route_reduce_kernel_id,
                            int dw1_kernel_id,
                            int bias_kernel_id);

void opus_moe_varlen_router_bwd(aiter_tensor_t& d_scores,
                                aiter_tensor_t& scores,
                                aiter_tensor_t& route_expert_ids,
                                aiter_tensor_t& token_route_offsets,
                                aiter_tensor_t& d_logits,
                                int kernel_id);

// Tensor binding for the first independently testable K1 family.  Outputs and
// workspace are supplied by Python so allocation never occurs inside a launch
// or HIP graph capture.
void opus_moe_down_bwd(aiter_tensor_t& d_out,
                       aiter_tensor_t& z,
                       aiter_tensor_t& w2,
                       aiter_tensor_t& scores,
                       aiter_tensor_t& sorted_token_ids,
                       aiter_tensor_t& sorted_expert_ids,
                       aiter_tensor_t& num_valid_ids,
                       aiter_tensor_t& d_scores_workspace,
                       aiter_tensor_t& d_z,
                       aiter_tensor_t& a_scaled,
                       aiter_tensor_t& d_scores,
                       int block_m,
                       int kernel_id);

void opus_moe_route_bwd(aiter_tensor_t& d_z,
                        aiter_tensor_t& w1,
                        aiter_tensor_t& sorted_token_ids,
                        aiter_tensor_t& sorted_expert_ids,
                        aiter_tensor_t& num_valid_ids,
                        aiter_tensor_t& expert_padded_offsets,
                        aiter_tensor_t& reverse_sorted,
                        aiter_tensor_t& d_x_route,
                        aiter_tensor_t& d_x,
                        int block_m,
                        int route_dx_kernel_id,
                        int route_reduce_kernel_id);

// Combined K4/K5 tensor binding.  Python owns output allocation; both kernels
// consume the same padded per-expert sorted-route intervals without redoing
// TopK or introducing a host synchronization.
void opus_moe_weight_bwd(aiter_tensor_t& x,
                         aiter_tensor_t& d_out,
                         aiter_tensor_t& d_z,
                         aiter_tensor_t& a_scaled,
                         aiter_tensor_t& sorted_token_ids,
                         aiter_tensor_t& num_valid_ids,
                         aiter_tensor_t& expert_padded_offsets,
                         aiter_tensor_t& d_w1,
                         aiter_tensor_t& d_w2,
                         int topk,
                         int block_m,
                         int dw1_kernel_id,
                         int dw2_kernel_id);

void opus_moe_dw1_bwd(aiter_tensor_t& x,
                      aiter_tensor_t& d_z,
                      aiter_tensor_t& sorted_token_ids,
                      aiter_tensor_t& num_valid_ids,
                      aiter_tensor_t& expert_padded_offsets,
                      aiter_tensor_t& d_w1,
                      int topk,
                      int block_m,
                      int kernel_id);

void opus_moe_dw2_bwd(aiter_tensor_t& d_out,
                      aiter_tensor_t& a_scaled,
                      aiter_tensor_t& sorted_token_ids,
                      aiter_tensor_t& num_valid_ids,
                      aiter_tensor_t& expert_padded_offsets,
                      aiter_tensor_t& d_w2,
                      int topk,
                      int block_m,
                      int kernel_id);

// Full K1--K5 launch entry.  Keeping the five dependent launches behind one
// extension boundary avoids Python/custom-op bubbles between kernel families
// while preserving the independently testable family entry points above.
void opus_moe_full_bwd(aiter_tensor_t& d_out,
                       aiter_tensor_t& x,
                       aiter_tensor_t& z,
                       aiter_tensor_t& w1,
                       aiter_tensor_t& w2,
                       aiter_tensor_t& scores,
                       aiter_tensor_t& sorted_token_ids,
                       aiter_tensor_t& sorted_expert_ids,
                       aiter_tensor_t& num_valid_ids,
                       aiter_tensor_t& reverse_sorted,
                       aiter_tensor_t& expert_padded_offsets,
                       aiter_tensor_t& d_scores_workspace,
                       aiter_tensor_t& d_z,
                       aiter_tensor_t& a_scaled,
                       aiter_tensor_t& d_scores,
                       aiter_tensor_t& d_x_route,
                       aiter_tensor_t& d_x,
                       aiter_tensor_t& d_w1,
                       aiter_tensor_t& d_w2,
                       int block_m,
                       int down_kernel_id,
                       int route_dx_kernel_id,
                       int route_reduce_kernel_id,
                       int dw1_kernel_id,
                       int dw2_kernel_id);

// Trusted full-chain entry for an internal training wrapper that has already
// validated all reusable tensors and established the current device/stream.
// It intentionally skips duplicate tensor/device checks and launch-error
// polling; user-facing APIs must call opus_moe_full_bwd above.
void opus_moe_full_bwd_trusted(aiter_tensor_t& d_out,
                               aiter_tensor_t& x,
                               aiter_tensor_t& z,
                               aiter_tensor_t& w1,
                               aiter_tensor_t& w2,
                               aiter_tensor_t& scores,
                               aiter_tensor_t& sorted_token_ids,
                               aiter_tensor_t& sorted_expert_ids,
                               aiter_tensor_t& num_valid_ids,
                               aiter_tensor_t& reverse_sorted,
                               aiter_tensor_t& expert_padded_offsets,
                               aiter_tensor_t& d_scores_workspace,
                               aiter_tensor_t& d_z,
                               aiter_tensor_t& a_scaled,
                               aiter_tensor_t& d_scores,
                               aiter_tensor_t& d_x_route,
                               aiter_tensor_t& d_x,
                               aiter_tensor_t& d_w1,
                               aiter_tensor_t& d_w2,
                               int block_m,
                               int down_kernel_id,
                               int route_dx_kernel_id,
                               int route_reduce_kernel_id,
                               int dw1_kernel_id,
                               int dw2_kernel_id);
