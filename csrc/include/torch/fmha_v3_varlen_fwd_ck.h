#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor> fmha_v3_varlen_fwd_ck(const at::Tensor& q,            // [total_q, hq, d]
                                              const at::Tensor& k,            // [total_k, hk, d]
                                              const at::Tensor& v,            // [total_k, hk, d]
                                              const at::Tensor& cu_seqlens_q, // [b+1]
                                              const at::Tensor& cu_seqlens_k, // [b+1]
                                              int max_seqlen_q,
                                              int max_seqlen_k,
                                              float softmax_scale,
                                              bool is_causal);
} // namespace torch_itfs
} // namespace aiter
