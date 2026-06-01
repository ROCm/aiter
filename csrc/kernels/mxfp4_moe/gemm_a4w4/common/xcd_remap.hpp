// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

#include "common/mxfp4_gemm_common.hpp"

namespace aiter::mxfp4_moe::gemm_common {

template <int NUM_XCDS = 8, int XCD_SWIZZLE = 0>
DEVICE_INLINE void remap_xcd_grouped(
    int pid_raw,
    int total_m_blocks,
    int num_n_blocks,
    int& m_block_idx,
    int& n_block_idx)
{
    static_assert(NUM_XCDS > 0, "NUM_XCDS must be positive");
    if constexpr (XCD_SWIZZLE == 0) {
        m_block_idx = pid_raw / num_n_blocks;
        n_block_idx = pid_raw % num_n_blocks;
        return;
    }

    const int total_wgs = total_m_blocks * num_n_blocks;
    const int q  = total_wgs / NUM_XCDS;
    const int r  = total_wgs % NUM_XCDS;
    const int xcd     = pid_raw % NUM_XCDS;
    const int in_xcd  = pid_raw / NUM_XCDS;
    const int clip    = (xcd < r) ? xcd : r;
    const int wgid    = xcd * q + clip + in_xcd;

    if constexpr (XCD_SWIZZLE == -1) {
        m_block_idx = wgid / num_n_blocks;
        n_block_idx = wgid % num_n_blocks;
        return;
    }

    static_assert(XCD_SWIZZLE > 0, "XCD_SWIZZLE must be 0, -1, or positive");
    const int num_wgid_in_group = XCD_SWIZZLE * num_n_blocks;
    const int group_id          = wgid / num_wgid_in_group;
    const int first_pid_m       = group_id * XCD_SWIZZLE;
    const int remaining_m       = total_m_blocks - first_pid_m;
    const int group_size_m      = (remaining_m < XCD_SWIZZLE) ? remaining_m : XCD_SWIZZLE;
    const int wgid_in_group     = wgid % num_wgid_in_group;
    m_block_idx = first_pid_m + (wgid_in_group % group_size_m);
    n_block_idx = wgid_in_group / group_size_m;
}

}  // namespace aiter::mxfp4_moe::gemm_common
