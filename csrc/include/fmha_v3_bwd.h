#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

std::vector<at::Tensor>
fmha_v3_bwd(const at::Tensor &dout, // [b, sq, hq, d]
        const at::Tensor &q,    // [b, sq, hq, d]
        const at::Tensor &k,    // [b, sk, hk, d]
        const at::Tensor &v,    // [b, sk, hk, d]
        const at::Tensor &out,  // [b, sq, hq, d]
        const at::Tensor &lse,  // [b, hq, sq]
        float p_dropout,
        float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool deterministic,
        bool is_v3_atomic_fp32,
        int how_v3_bf16_cvt,
        std::optional<at::Tensor> dq,                 // [b, sq, hq, d]
        std::optional<at::Tensor> dk,                 // [b, sk, hk, d]
        std::optional<at::Tensor> dv,                 // [b, sk, hk, d]
        std::optional<const at::Tensor> alibi_slopes, // [hq] or [b, hq]
        std::optional<const at::Tensor> rng_state,
        std::optional<at::Generator> gen);
