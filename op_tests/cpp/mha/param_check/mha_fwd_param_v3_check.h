// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "mha_fwd.h"

enum arch_enum {
    gfx942 = 0,
    gfx950 = 1
};

bool mha_fwd_v3_check(const arch_enum gfx_version,
                    aiter::mha_fwd_args args,
                    const ck_tile::stream_config& stream_config,
                    std::string q_dtype_str,
                    bool is_group_mode,
                    mask_enum mask_type,
                    bias_enum bias_type,
                    bool has_lse,
                    bool use_ext_asm,
                    int how_v3_bf16_cvt                = 1,
                    const void* seqstart_q_padding_ptr = nullptr,
                    const void* seqstart_k_padding_ptr = nullptr);

float fmha_fwd_v3_gfx942_check(aiter::mha_fwd_traits t,
                  aiter::mha_fwd_args a,
                  const ck_tile::stream_config& s,
                  const void* seqstart_q_padding_ptr = nullptr,
                  const void* seqstart_k_padding_ptr = nullptr);

float fmha_fwd_v3_gfx950_check(aiter::mha_fwd_traits t,
                  aiter::mha_fwd_args a,
                  const ck_tile::stream_config& s,
                  const void* seqstart_q_padding_ptr = nullptr,
                  const void* seqstart_k_padding_ptr = nullptr);
