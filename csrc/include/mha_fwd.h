#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

std::vector<at::Tensor>
mha_fwd(at::Tensor &q, // [b, sq, hq, d]
        at::Tensor &k, // [b, sk, hk, d]
        at::Tensor &v, // [b, sk, hk, d]
        float p_dropout,
        float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool return_softmax_lse,
        bool return_dropout_randval,
        std::optional<at::Tensor> out,          // [b, sq, hq, d]
        std::optional<at::Tensor> alibi_slopes, // [hq] or [b, hq]
        std::optional<at::Generator> gen);
