// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <hip/hip_bfloat16.h>

namespace opus_moe
{

constexpr int kStage2KidAuto = -1;

} // namespace opus_moe

struct opus_moe_stage2_route_reduce_kargs
{
    const uint8_t* __restrict__ route_out;
    hip_bfloat16* __restrict__ out_bf16;

    int token_num;
    int topk;
    int model_dim;

    int64_t stride_o_t;
    int64_t stride_route_out_t;  // BF16 route_out row stride, in bf16 elements.
    int route_out_fp8;           // MXFP8 route_out reduce: read fp8 + per-8col e8m0 scale.
    int64_t route_out_row_bytes; // FP8 route_out row stride bytes (scale at row+model_dim).
};

static __device__ __forceinline__ int opus_moe_token_id(int32_t packed)
{
    return packed & 0x00ffffff;
}

static __device__ __forceinline__ int opus_moe_topk_slot(int32_t packed)
{
    return static_cast<uint32_t>(packed) >> 24;
}
