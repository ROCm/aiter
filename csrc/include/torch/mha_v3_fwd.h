#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor> fmha_v3_fwd(at::Tensor &q, // [b, sq, hq, d]
                                    const at::Tensor &k, // [b, sk, hk, d]
                                    const at::Tensor &v, // [b, sk, hk, d_v]
                                    float p_dropout,
                                    float softmax_scale,
                                    bool is_causal,
                                    int window_size_left,
                                    int window_size_right,
                                    bool return_softmax_lse,
                                    bool return_dropout_randval,
                                    std::optional<at::Tensor> out_,          // [b, sq, hq, d_v]
                                    std::optional<const at::Tensor> bias_,   // [sq, sk]
                                    std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
                                    std::optional<at::Generator> gen_)
} // namespace torch_itfs
} // namespace aiter
