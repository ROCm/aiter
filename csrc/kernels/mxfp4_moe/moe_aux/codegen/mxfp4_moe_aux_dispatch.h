// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace aiter::mxfp4_moe::aux_dispatch {

// ── Launch-config constants (shape-independent; baked into codegen'd bodies) ──
constexpr int kNCtasSort              = 512;
constexpr int kThreadsSort            = 1024;
constexpr int kNCtasScales            = 512;
constexpr int kThreadsScales          = 1024;
constexpr int kThreadsScatterReduce   = 128;
constexpr int kColsPerThread          = 8;
constexpr int kColsPerThreadQ         = 8;    // mxfp4-input reduce: 8 fp4 = one u32 load
constexpr int kThreadsScatterReduceQ  = 128;  // mxfp4-input reduce CTA size
constexpr int kSplitSortCtas          = 16;
constexpr int kInlineQuantZeroInitCtas = 128;
constexpr int kBK                     = 256;  // sort_scales K-tile

// ── Per-entry function-pointer types (one per aux launch entry) ──────────────
// bf16 / uint8 / fp4 buffers are void*; the wrapper body reinterpret_casts them.

using SortQuantFn = void (*)(
    hipStream_t stream,
    int            M,
    const void*    a_input,
    const int32_t* topk_ids,
    const float*   topk_weight,
    int32_t*       sorted_token_ids,
    int32_t*       sorted_expert_ids,
    int32_t*       cumsum,
    int32_t*       reverse_sorted,
    float*         sorted_weights,
    void*          a_quant,
    void*          a_scale,
    int32_t*       m_indices,
    void*          bf16_zero_ptr);

using Sort3StageFn = void (*)(
    hipStream_t stream,
    int            M,
    const int32_t* topk_ids,
    const float*   topk_weight,
    int32_t*       sorted_token_ids,
    int32_t*       sorted_expert_ids,
    int32_t*       cumsum,
    int32_t*       reverse_sorted,
    float*         sorted_weights,
    int32_t*       m_indices,
    int32_t*       block_offsets,
    int32_t*       real_counts);

using SortOnlyZiFn = void (*)(
    hipStream_t stream,
    int            M,
    const int32_t* topk_ids,
    const float*   topk_weight,
    int32_t*       sorted_token_ids,
    int32_t*       sorted_expert_ids,
    int32_t*       cumsum,
    int32_t*       reverse_sorted,
    float*         sorted_weights,
    int32_t*       m_indices,
    void*          bf16_zero_ptr,
    void*          bf16_zero_ws_ptr,
    long long      workspace_bytes);

using SortOnlyFn = void (*)(
    hipStream_t stream,
    int            M,
    const int32_t* topk_ids,
    const float*   topk_weight,
    int32_t*       sorted_token_ids,
    int32_t*       sorted_expert_ids,
    int32_t*       cumsum,
    int32_t*       reverse_sorted,
    float*         sorted_weights,
    int32_t*       m_indices);

using QuantFn = void (*)(
    hipStream_t stream,
    int            M,
    const void*    a_input,
    void*          a_quant,
    void*          a_scale,
    void*          bf16_zero_ptr);

using SortScalesFn = void (*)(
    hipStream_t stream,
    int            M,
    int            max_sorted,
    void*          a_scale,
    const int32_t* sorted_token_ids,
    const int32_t* cumsum,
    void*          a_scale_sorted_shuffled);

using ScatterReduceFn = void (*)(
    hipStream_t stream,
    int            M,
    const void*    flat_out,
    const int32_t* reverse_sorted,
    const float*   sorted_weights,
    void*          out);

using ScatterReduceQFn = void (*)(
    hipStream_t stream,
    int            M,
    const void*    flat_out_q,
    const void*    flat_out_scale,
    const int32_t* reverse_sorted,
    const float*   sorted_weights,
    void*          out);

}  // namespace aiter::mxfp4_moe::aux_dispatch
