// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd_v3.hpp"

template <>
float fmha_fwd_v3_dispatch<type_tag<ck_tile::fmha_fwd_v3_args::data_type_enum::fp16, false>>(
    const ck_tile::fmha_fwd_v3_args& args, const ck_tile::stream_config& config)
{
    return ck_tile::launch<ck_tile::get_kernel_t<FmhaFwdFp16, true, false>>(args, config);
}
