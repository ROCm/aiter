// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <hip/hip_runtime.h>
#include <opus/opus.hpp>
using B = opus::dtypes::bf16;
struct FusedDsv41Args
{
    const B* q;
    const uint8_t* pool;
    const int64_t* prefix;
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
};
hipError_t launch_fused_dsv41_mxfp8(const FusedDsv41Args*, int, hipStream_t);
