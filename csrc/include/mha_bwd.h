#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include "fmha_bwd.hpp"
#include "mask.hpp"

struct fmha_bwd_traits_all: public fmha_bwd_traits
{
    fmha_bwd_traits_all(const mask_info &mask,
        std::string dtype,
        int head_size,
        bool has_dropout,
        bool enable_alibi,
        bool deterministic,
        bool use_ext_asm,
        bool is_v3_atomic_fp32,
        int how_v3_bf16_cvt): fmha_bwd_traits{head_size,
            head_size,
            dtype,
            false, // is_group_mode
            mask.type,
            enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
            false,    // has_dbias
            has_dropout,
            false, // s_randval
            deterministic}, 
            use_ext_asm{use_ext_asm},
            is_v3_atomic_fp32{is_v3_atomic_fp32},
            how_v3_bf16_cvt{how_v3_bf16_cvt} {}
    bool use_ext_asm;
    bool is_v3_atomic_fp32;
    int how_v3_bf16_cvt;
};

float fmha_bwd_v3(fmha_bwd_traits_all, fmha_bwd_args, const ck_tile::stream_config&);

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout, // [b, sq, hq, d]
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
        std::optional<at::Tensor> dq,                 // [b, sq, hq, d]
        std::optional<at::Tensor> dk,                 // [b, sk, hk, d]
        std::optional<at::Tensor> dv,                 // [b, sk, hk, d]
        std::optional<const at::Tensor> alibi_slopes, // [hq] or [b, hq]
        std::optional<const at::Tensor> rng_state,
        std::optional<at::Generator> gen);
