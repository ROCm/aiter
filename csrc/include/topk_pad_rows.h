#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <hip/hip_runtime.h>
#include <optional>

namespace aiter {

// A row at or past *num_valid_rows is padding (a captured graph's or DP's pad
// rows): a top-k routing kernel writes its routed columns as id -1, which
// expert-parallel dispatch and MoE sorting skip, and weight 0. Read on device,
// so a replayed graph follows its step's count. nullptr: every row is real.
__device__ __forceinline__ bool is_pad_row(const int* num_valid_rows, int row)
{
    return num_valid_rows != nullptr && row >= *num_valid_rows;
}

// Writes a pad row's `topk` routed columns, lane-strided over `lanes` threads.
__device__ __forceinline__ void
write_pad_row(float* weights, int* ids, int topk, int lane, int lanes)
{
    for(int k = lane; k < topk; k += lanes)
    {
        weights[k] = 0.0f;
        ids[k]     = -1;
    }
}

// The device count a routing op's optional `num_valid_rows` names, checked:
// an int32 tensor whose first element is the count. nullptr when absent.
inline const int* num_valid_rows_ptr(const std::optional<aiter_tensor_t>& num_valid_rows)
{
    if(!num_valid_rows.has_value())
        return nullptr;
    AITER_CHECK(num_valid_rows->dtype() == AITER_DTYPE_i32 && num_valid_rows->numel() >= 1,
                "num_valid_rows must be an int32 tensor of at least one element");
    return reinterpret_cast<const int*>(num_valid_rows->data_ptr());
}

} // namespace aiter
