#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include "fmha_bwd.hpp"
#include "mask.hpp"

namespace aiter {
struct mha_bwd_traits : public fmha_bwd_traits
{
    mha_bwd_traits(int head_size_q,
                   int head_size_v,
                   std::string dtype,
                   bool is_group_mode,
                   mask_enum mask_type,
                   bias_enum bias_type,
                   bool has_dbias,
                   bool has_dropout,
                   bool is_store_randval,
                   bool deterministic,
                   bool use_ext_asm,
                   bool is_v3_atomic_fp32,
                   int how_v3_bf16_cvt)
        : fmha_bwd_traits{head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          mask_type,
                          bias_type,
                          has_dbias,
                          has_dropout,
                          is_store_randval,
                          deterministic},
          use_ext_asm(use_ext_asm),
          is_v3_atomic_fp32(is_v3_atomic_fp32),
          how_v3_bf16_cvt(how_v3_bf16_cvt)
    {
    }
    bool use_ext_asm;
    bool is_v3_atomic_fp32;
    int how_v3_bf16_cvt;
};

using mha_bwd_args = fmha_bwd_args;

__attribute__((visibility("default")))
float mha_bwd(mha_bwd_args args,
              const ck_tile::stream_config& stream_config,
              std::string q_dtype_str,
              bool is_group_mode,
              mask_enum mask_type,
              bias_enum bias_type,
              bool has_dbias,
              bool is_store_randval,
              bool deterministic,
              bool use_ext_asm,
              bool is_v3_atomic_fp32,
              int how_v3_bf16_cvt);

float fmha_bwd_v3(mha_bwd_traits t, mha_bwd_args a, const ck_tile::stream_config& s);

} // namespace aiter
