#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include "fmha_fwd.hpp"
#include "mask.hpp"

namespace aiter {
struct mha_fwd_traits : public fmha_fwd_traits {
  mha_fwd_traits(int head_size_q, int head_size_v, std::string dtype,
                 bool is_group_mode, bool has_logits_soft_cap,
                 mask_enum mask_type, bias_enum bias_type, bool has_lse,
                 bool has_dropout, bool use_ext_asm,
                 bool skip_min_seqlen_q)
      : fmha_fwd_traits{head_size_q,
                        head_size_v,
                        dtype,
                        is_group_mode,
                        true, // is_v_rowmajor
                        has_logits_soft_cap,
                        mask_type,
                        bias_type,
                        has_lse,
                        has_dropout,
                        false, // do_fp8_static_quant
                        skip_min_seqlen_q},
        use_ext_asm(use_ext_asm) {}
  bool use_ext_asm;
};

struct mha_fwd_splitkv_traits : public fmha_fwd_splitkv_traits {
  mha_fwd_splitkv_traits(int head_size_q, int head_size_v, std::string dtype,
                         bool is_group_mode, bool has_logits_soft_cap,
                         mask_enum mask_type, bias_enum bias_type, bool has_lse)
      : fmha_fwd_splitkv_traits{head_size_q,
                                head_size_v,
                                dtype,
                                is_group_mode,
                                true, // is_v_rowmajor
                                has_logits_soft_cap,
                                mask_type,
                                bias_type,
                                has_lse,
                                false} // do_fp8_static_quant
  {}
};

using mha_fwd_args = fmha_fwd_args;
using mha_fwd_splitkv_args = fmha_fwd_splitkv_args;
using mha_batch_prefill_args = fmha_batch_prefill_args;

float mha_fwd(mha_fwd_args args, const ck_tile::stream_config &stream_config,
              std::string q_dtype_str, bool is_group_mode, mask_enum mask_type,
              bias_enum bias_type, bool has_lse, bool use_ext_asm);

float mha_fwd_splitkv(mha_fwd_splitkv_args args,
                      const ck_tile::stream_config &stream_config,
                      std::string q_dtype_str, bool is_group_mode,
                      mask_enum mask_type, bias_enum bias_type, bool has_lse);

float mha_batch_prefill(mha_batch_prefill_args args,
                        const ck_tile::stream_config &stream_config,
                        std::string q_dtype_str, bool is_group_mode,
                        mask_enum mask_type, bias_enum bias_type, bool has_lse,
                        bool use_ext_asm);

float fmha_fwd_v3(mha_fwd_traits t, fmha_fwd_args a,
                  const ck_tile::stream_config &s);

} // namespace aiter
