// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// #include "mha_fwd.h"
#include "mha_bwd.h"

enum arch_enum {
    gfx942 = 0,
    gfx950 = 1
};

bool mha_bwd_v3_check(const arch_enum gfx_version,
            fmha_bwd_args args,
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
            int how_v3_bf16_cvt,
            const void* seqlen_q_padded = nullptr,
            const void* seqlen_k_padded = nullptr);

float fmha_bwd_v3_gfx942_check(aiter::mha_bwd_traits t,
                  fmha_bwd_args a,
                  const ck_tile::stream_config& s,
                  const void* seqlen_q_padded = nullptr,
                  const void* seqlen_k_padded = nullptr);

float fmha_bwd_v3_gfx950_check(aiter::mha_bwd_traits t,
                  fmha_bwd_args a,
                  const ck_tile::stream_config& s,
                  const void* seqlen_q_padded = nullptr,
                  const void* seqlen_k_padded = nullptr);
