// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950-specific Opus MoE dispatch implementations.
#pragma once

#include "../opus_moe_common.cuh"
#include "aiter_hip_common.h"

constexpr int kOpusMoeStage2RouteOutputReduceBf16BlockN =
    opus_moe::kStage2RouteOutputReduceBf16BlockN;
constexpr int kOpusMoeStage2RouteOutputReduceDefaultBlockN =
    opus_moe::kStage2RouteOutputReduceDefaultBlockN;
constexpr int kOpusMoeStage2RouteOutputReduceDefaultThreads =
    opus_moe::kStage2RouteOutputReduceDefaultThreads;

template<int BLOCK_N, int BLOCK_THREADS, int TOPK = 0, bool ROUTE_FP8 = false>
__global__ __launch_bounds__(BLOCK_THREADS, 4) void
opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950(
    opus_moe_stage2_route_reduce_kargs kargs);

inline int opus_moe_stage2_reduce_token_slot_route_output_select_block_n(
    int model_dim,
    int requested_block_n)
{
    if(requested_block_n > 0)
        return requested_block_n;
    const int auto_block_n = opus_moe::stage2_a8w4_route_reduce_auto_block_n(model_dim);
    return auto_block_n > 0 ? auto_block_n : kOpusMoeStage2RouteOutputReduceDefaultBlockN;
}

template<int BLOCK_N, int BLOCK_THREADS, int TOPK, bool ROUTE_FP8>
inline void opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950(
    const opus_moe_stage2_route_reduce_kargs& kargs,
    dim3 grid,
    hipStream_t stream)
{
    opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950<
        BLOCK_N,
        BLOCK_THREADS,
        TOPK,
        ROUTE_FP8><<<grid, dim3(BLOCK_THREADS), 0, stream>>>(kargs);
}

template<int BLOCK_N, int BLOCK_THREADS, int TOPK>
inline void opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950(
    const opus_moe_stage2_route_reduce_kargs& kargs,
    dim3 grid,
    hipStream_t stream)
{
    if(kargs.route_out_fp8)
    {
        opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950<
            BLOCK_N,
            BLOCK_THREADS,
            TOPK,
            true>(kargs, grid, stream);
    }
    else
    {
        opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950<
            BLOCK_N,
            BLOCK_THREADS,
            TOPK,
            false>(kargs, grid, stream);
    }
}

#include "opus_moe_stage2_a8w4_manifest.h"

// Dispatch on block_n with the topk known at compile time (TOPK).
template<int TOPK>
inline void opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950(
    const opus_moe_stage2_route_reduce_kargs& kargs,
    dim3 grid,
    hipStream_t stream,
    int block_n)
{
    switch(block_n)
    {
    case kOpusMoeStage2RouteOutputReduceBf16BlockN:
        opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950<
            kOpusMoeStage2RouteOutputReduceBf16BlockN,
            kOpusMoeStage2RouteOutputReduceDefaultThreads,
            TOPK>(kargs, grid, stream);
        break;
    case kOpusMoeStage2RouteOutputReduceDefaultBlockN:
        opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950<
            kOpusMoeStage2RouteOutputReduceDefaultBlockN,
            kOpusMoeStage2RouteOutputReduceDefaultThreads,
            TOPK>(kargs, grid, stream);
        break;
    default:
        if(opus_moe_stage2_a8w4_route_reduce_dispatch_generated_gfx950<TOPK>(
               kargs, grid, stream, block_n))
            break;
        AITER_CHECK(false,
                    "unsupported Opus MoE route-output reduce block_n=",
                    block_n);
    }
}

inline void opus_moe_stage2_reduce_token_slot_route_output_launch_gfx950(
    const opus_moe_stage2_route_reduce_kargs& kargs,
    hipStream_t stream,
    int requested_block_n)
{
    const int block_n = opus_moe_stage2_reduce_token_slot_route_output_select_block_n(
        kargs.model_dim, requested_block_n);
    dim3 grid(kargs.token_num, (kargs.model_dim + block_n - 1) / block_n, 1);
    // Specialize common topk values; TOPK=0 remains the runtime fallback.
    switch(kargs.topk)
    {
    case 4:
        opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950<4>(
            kargs, grid, stream, block_n);
        break;
    case 6:
        opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950<6>(
            kargs, grid, stream, block_n);
        break;
    case 8:
        opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950<8>(
            kargs, grid, stream, block_n);
        break;
    default:
        opus_moe_stage2_reduce_token_slot_route_output_dispatch_block_n_gfx950<0>(
            kargs, grid, stream, block_n);
        break;
    }
}
