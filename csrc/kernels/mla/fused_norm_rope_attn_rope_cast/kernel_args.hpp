// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "cache_layout.hpp"
#include <hip/hip_runtime.h>
#include <opus/opus.hpp>
using B = opus::dtypes::bf16;
struct FusedNormRopeAttnRopeCastArgs
{
    const B* q;
    const uint8_t* q_scale;
    const uint8_t* pool;
    const void* prefix;
    const int* prefix_indptr;
    const B* extend_kv;
    const int *extend, *extend_indptr;
    const void* positions;
    const B *cos, *sin;
    const float* sink;
    uint8_t *out, *scales;
    float *lse, *max_logits, *debug_out;
    int tokens, heads, extend_rows, prefix_capacity, extend_capacity, positions64;
    int64_t pool_bytes;
    float softmax_scale;
    bool fp8_qk;
    bool q_norm;
    float norm_eps;
    Fp8CacheLayout cache;
    template <int PageSize>
    __device__ __forceinline__ const void* prefix_at(int begin) const
    {
        return PageSize ? static_cast<const void*>(static_cast<const int*>(prefix) + begin)
                        : static_cast<const void*>(static_cast<const int64_t*>(prefix) + begin);
    }
};
hipError_t
launch_fused_norm_rope_attn_rope_cast(const FusedNormRopeAttnRopeCastArgs*, int, hipStream_t);
