// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace aiter::mxfp4_moe::dispatch {

using Gemm1CshuffleFn = void (*)(
    hipStream_t stream,
    const void*       A_q,
    const void*       A_scale,
    const void*       B_q,
    const void*       B_scale,
    const int32_t*    sorted_expert_ids,
    const int32_t*    cumsum,
    const int32_t*    m_indices,
    int               n_tokens,
    void*             A_q_out,
    void*             A_scale_out,
    const void*       hidden_ptr);

using Gemm2AtomicFn = void (*)(
    hipStream_t stream,
    const void*       A_q,
    const void*       A_scale,
    const void*       B_q,
    const void*       B_scale,
    const int32_t*    sorted_expert_ids,
    const int32_t*    cumsum,
    const int32_t*    sorted_token_ids,
    const float*      sorted_weights,
    int               M,
    void*             bf16_out);

using Gemm2NonatomicFn = void (*)(
    hipStream_t stream,
    const void*       A_q,
    const void*       A_scale,
    const void*       B_q,
    const void*       B_scale,
    const int32_t*    sorted_expert_ids,
    const int32_t*    cumsum,
    int               max_sorted,
    void*             bf16_out);

}  // namespace aiter::mxfp4_moe::dispatch
