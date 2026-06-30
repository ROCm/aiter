// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <hip/hip_bfloat16.h>

#include "opus_moe_stage2_a8w4_meta.h"

namespace opus_moe
{

constexpr int kStage2KidAuto = -1;
constexpr int kStage2KidBf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast = 1;

constexpr bool stage2_bf16_kid_is_valid(int kid)
{
    return kid == kStage2KidBf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast;
}

constexpr const char* stage2_bf16_kid_name(int kid)
{
    switch(kid)
    {
    case kStage2KidBf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast:
        return "bf16_gemmstyle256x256x64_token_slot_route_out_no_oob_nfast";
    default:
        return "unknown";
    }
}

} // namespace opus_moe

struct opus_moe_stage2_bf16_kargs
{
    const hip_bfloat16* __restrict__ inter_states;
    const hip_bfloat16* __restrict__ w2;
    const int32_t* __restrict__ sorted_token_ids;
    const float* __restrict__ sorted_weights;
    const int32_t* __restrict__ sorted_expert_ids;
    const int32_t* __restrict__ num_valid_ids;
    hip_bfloat16* __restrict__ route_out_bf16;

    int token_num;
    int topk;
    int num_experts;
    int model_dim;
    int inter_dim;
    int block_m;

    int64_t stride_a_t;
    int64_t stride_a_k;
    int64_t stride_w_e;
    int64_t stride_w_h;
    int64_t stride_route_out_t;
};

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

struct opus_moe_stage2_a8w4_kargs
{
    const uint8_t* __restrict__ inter_states_fp8;
    const uint8_t* __restrict__ w2_fp4;
    const uint8_t* __restrict__ a2_scale_e8m0;
    const uint8_t* __restrict__ w2_scale_e8m0;
    const int32_t* __restrict__ sorted_token_ids;
    const float* __restrict__ sorted_weights;
    const int32_t* __restrict__ sorted_expert_ids;
    const int32_t* __restrict__ num_valid_ids;
    hip_bfloat16* __restrict__ out_bf16;

    int64_t stride_a_t;
    int64_t stride_a_k;
    int64_t stride_w_e;
    int64_t stride_a_scale_route;
    int64_t stride_w_scale_row;
    int64_t stride_o_t;

    int token_num;
    int topk;
    int num_experts;
    int model_dim;
    int sorted_blocks;
    int route_out_fp8;          // Runtime guard for the MXFP8 route-out path.
    int64_t route_out_row_bytes;  // fp8 route_out row stride bytes (= model_dim + model_dim/8).
};

static __device__ __forceinline__ int opus_moe_token_id(int32_t packed)
{
    return packed & 0x00ffffff;
}

static __device__ __forceinline__ int opus_moe_topk_slot(int32_t packed)
{
    return static_cast<uint32_t>(packed) >> 24;
}
