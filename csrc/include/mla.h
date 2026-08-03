// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>

union MlaWorkInfo
{
    struct
    {
        int32_t batch_idx;
        int32_t partial_qo_loc;
        int32_t qo_start;
        int32_t qo_end;
        int32_t kv_start;
        int32_t kv_end;
        int32_t kv_offset;
        int32_t padding[1];
    };
    uint32_t u32All[8];
};
constexpr size_t kSizeMlaWorkInfoInDw = sizeof(MlaWorkInfo) / sizeof(uint32_t);
static_assert(kSizeMlaWorkInfoInDw == 8);

union MlaPartialTileInfo
{
    struct
    {
        int32_t q_start;
        int32_t q_end;
    };
    uint32_t u32All[2];
};
constexpr size_t kSizeMlaPartialTileInfoInDw = sizeof(MlaPartialTileInfo) / sizeof(uint32_t);
static_assert(kSizeMlaPartialTileInfoInDw == 2);
