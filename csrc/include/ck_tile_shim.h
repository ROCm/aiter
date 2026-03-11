#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Minimal shim providing ck_tile:: types when compiling without the full
// Composable Kernel dependency (ONLY_FAV3==1).

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cassert>

namespace ck_tile {

using index_t      = int32_t;
using long_index_t = int64_t;

struct stream_config
{
    hipStream_t stream_id_ = nullptr;
    bool time_kernel_      = false;
    int log_level_         = 0;
    int cold_niters_       = 3;
    int nrepeat_           = 10;
    bool is_gpu_timer_     = true;
    bool flush_cache_      = false;
    int rotating_count_    = 1;
};

template <typename T>
constexpr T log2e_v = static_cast<T>(1.4426950408889634);

inline int get_warp_size() { return 64; }

template <typename... Callables>
float launch_kernel(const stream_config& s, Callables&&... callables)
{
    (callables(s), ...);
    return 0;
}

} // namespace ck_tile
