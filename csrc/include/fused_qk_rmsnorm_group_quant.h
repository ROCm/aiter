// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

void fused_qk_rmsnorm_group_quant(std::optional<aiter_tensor_t> q_out_quantized,
                                  std::optional<aiter_tensor_t> q_out_scale,
                                  std::optional<aiter_tensor_t> q,
                                  std::optional<aiter_tensor_t> q_weight,
                                  std::optional<double> q_epsilon,
                                  std::optional<aiter_tensor_t> q_out_unquantized,
                                  std::optional<aiter_tensor_t> k_out,
                                  std::optional<aiter_tensor_t> q_res_out,
                                  std::optional<aiter_tensor_t> k,
                                  std::optional<aiter_tensor_t> k_weight,
                                  std::optional<double> k_epsilon,
                                  std::optional<aiter_tensor_t> q_residual,
                                  int64_t group_size,
                                  bool transpose_scale,
                                  bool gemma_norm,
                                  double amax_floor = 0.0);

// fused_qk_rmsnorm_group_quant's MXFP8 dual RMSNorm whose bf16 kv row then runs straight
// through the KV half of ATOM's rope_quant_window (RoPE tail, MXFP8 per 32, qat / values +
// scales / window ring sinks). kv_out optionally keeps the normed bf16 row.
void fused_qk_rmsnorm_mxfp8_kv_rope_quant(aiter_tensor_t& q_out,
                                          aiter_tensor_t& q_out_scale,
                                          aiter_tensor_t& q,
                                          aiter_tensor_t& q_weight,
                                          double q_epsilon,
                                          aiter_tensor_t& kv,
                                          aiter_tensor_t& kv_weight,
                                          double kv_epsilon,
                                          aiter_tensor_t& cos,
                                          aiter_tensor_t& sin,
                                          aiter_tensor_t& positions,
                                          int64_t rope_dim,
                                          double amax_floor,
                                          std::optional<aiter_tensor_t> kv_out,
                                          std::optional<aiter_tensor_t> qat,
                                          std::optional<aiter_tensor_t> values,
                                          std::optional<aiter_tensor_t> scales,
                                          std::optional<aiter_tensor_t> pool,
                                          std::optional<aiter_tensor_t> batch_ids,
                                          std::optional<aiter_tensor_t> cu_seqlens_q,
                                          std::optional<aiter_tensor_t> slots,
                                          int64_t ring_start,
                                          int64_t ring_slots,
                                          int64_t slot_rows,
                                          int64_t ring_stride,
                                          int64_t run_rows);

void fused_qk_rmsnorm_per_token_quant(aiter_tensor_t& q_out_quantized,
                                      aiter_tensor_t& q_out_scale,
                                      aiter_tensor_t& q,
                                      aiter_tensor_t& q_weight,
                                      double q_epsilon,
                                      std::optional<aiter_tensor_t> q_out_unquantized,
                                      std::optional<aiter_tensor_t> k_out,
                                      std::optional<aiter_tensor_t> q_res_out,
                                      std::optional<aiter_tensor_t> k,
                                      std::optional<aiter_tensor_t> k_weight,
                                      std::optional<double> k_epsilon,
                                      std::optional<aiter_tensor_t> q_residual,
                                      bool gemma_norm);

} // namespace aiter
