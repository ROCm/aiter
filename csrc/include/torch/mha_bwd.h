#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {
void mha_bwd(
                                at::Tensor &dq,
                                at::Tensor &dk,
                                at::Tensor &dv,
                                at::Tensor &softmax_d,
                                const at::Tensor& dout, // [b, sq, hq, d]
                                const at::Tensor& q,    // [b, sq, hq, d]
                                const at::Tensor& k,    // [b, sk, hk, d]
                                const at::Tensor& v,    // [b, sk, hk, d]
                                const at::Tensor& out,  // [b, sq, hq, d]
                                const at::Tensor& lse,  // [b, hq, sq]
                                float p_dropout,
                                float softmax_scale,
                                bool is_causal,
                                int window_size_left,
                                int window_size_right,
                                bool deterministic,
                                std::optional<at::Tensor> dq_,                 // [b, sq, hq, d]
                                std::optional<at::Tensor> dk_,                 // [b, sk, hk, d]
                                std::optional<at::Tensor> dv_,                 // [b, sk, hk, d]
                                std::optional<at::Tensor> dbias_,             // [sq, sk]
                                std::optional<const at::Tensor> bias_,        // [sq, sk]
                                std::optional<const at::Tensor> alibi_slopes, // [hq] or [b, hq]
                                std::optional<const at::Tensor> rng_state,
                                std::optional<at::Generator> gen);
} // namespace torch_itfs
} // namespace aiter
