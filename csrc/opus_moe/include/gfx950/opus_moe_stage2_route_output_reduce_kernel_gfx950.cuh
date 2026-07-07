// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../opus_moe_common.cuh"
#include "opus_moe_stage2_utils_gfx950.cuh"

#include <cstdint>

#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"

static __device__ __forceinline__ void
opus_moe_stage2_route_reduce_accum_bf16x4(float* acc, int base, uint64_t packed)
{
    const hip_bfloat16 v0 =
        opus_moe_gfx950_bf16_from_bits(static_cast<uint16_t>(packed));
    const hip_bfloat16 v1 =
        opus_moe_gfx950_bf16_from_bits(static_cast<uint16_t>(packed >> 16));
    const hip_bfloat16 v2 =
        opus_moe_gfx950_bf16_from_bits(static_cast<uint16_t>(packed >> 32));
    const hip_bfloat16 v3 =
        opus_moe_gfx950_bf16_from_bits(static_cast<uint16_t>(packed >> 48));
    acc[base + 0] += static_cast<float>(v0);
    acc[base + 1] += static_cast<float>(v1);
    acc[base + 2] += static_cast<float>(v2);
    acc[base + 3] += static_cast<float>(v3);
}

static __device__ __forceinline__ uint64_t
opus_moe_stage2_route_reduce_pack_bf16x4(const float* acc, int base)
{
    const uint32_t packed01 =
        opus_moe_gfx950_cvt_pk_bf16_f32(acc[base + 0], acc[base + 1]);
    const uint32_t packed23 =
        opus_moe_gfx950_cvt_pk_bf16_f32(acc[base + 2], acc[base + 3]);
    return static_cast<uint64_t>(packed01) | (static_cast<uint64_t>(packed23) << 32);
}

struct OpusMoeStage2RouteReduceMxFp8Group
{
    uint64_t data;
    float scale;
};

struct OpusMoeStage2RouteReduceMxFp8View
{
    const uint8_t* __restrict__ ptr;
    int64_t row_stride_bytes;
    int scale_off;

    __device__ __forceinline__ OpusMoeStage2RouteReduceMxFp8Group
    load_group(int route_row, int col) const
    {
        const uint8_t* row = ptr + static_cast<int64_t>(route_row) * row_stride_bytes;
        return {*reinterpret_cast<const uint64_t*>(row + col),
                opus_moe_gfx950_e8m0_to_float_scale(row[scale_off + (col >> 3)])};
    }
};

struct OpusMoeStage2RouteReduceBf16RouteView
{
    const hip_bfloat16* __restrict__ route_out;
    int64_t route_stride;

    __device__ __forceinline__ uint64_t load_x4(int route_row, int col) const
    {
        return *reinterpret_cast<const uint64_t*>(
            route_out + static_cast<int64_t>(route_row) * route_stride + col);
    }

    __device__ __forceinline__ hip_bfloat16 load(int route_row, int col) const
    {
        return route_out[static_cast<int64_t>(route_row) * route_stride + col];
    }
};

struct OpusMoeStage2RouteReduceBf16OutView
{
    hip_bfloat16* __restrict__ out;
    int64_t out_stride;

    __device__ __forceinline__ void store_x4(int token, int col, uint64_t packed) const
    {
        *reinterpret_cast<uint64_t*>(
            out + static_cast<int64_t>(token) * out_stride + col) = packed;
    }

    __device__ __forceinline__ void store(int token, int col, hip_bfloat16 value) const
    {
        out[static_cast<int64_t>(token) * out_stride + col] = value;
    }
};

static __device__ __forceinline__ opus::fp32x4_t
opus_moe_stage2_route_reduce_dequant_fp8x4(uint32_t packed, float scale)
{
    const auto values =
        opus::fp8_to_fp32_packed_x4(__builtin_bit_cast(opus::fp8x4_t, packed));
    return values * opus::fp32x4_t{scale, scale, scale, scale};
}
#endif

// ROUTE_FP8 selects the packed MXFP8 path at compile time. Unlike decode, this
// reduce kernel is small enough that specializing the output format trims the
// unused bf16 path without increasing VGPR pressure.
// TOPK > 0  -> slot loop bound is a compile-time constant so the whole loop
//              unrolls and issues the route loads before the final reduction.
// TOPK == 0 -> fallback to the runtime kargs.topk loop.
template<int BLOCK_N, int BLOCK_THREADS, int TOPK = 0, bool ROUTE_FP8 = false>
__global__ __launch_bounds__(BLOCK_THREADS, 4) void
opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950(opus_moe_stage2_route_reduce_kargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    static_assert(BLOCK_N % BLOCK_THREADS == 0);
    constexpr int elems_per_thread = BLOCK_N / BLOCK_THREADS;
    const int token = static_cast<int>(opus::block_id_x());
    const int col_base =
        static_cast<int>(opus::block_id_y()) * BLOCK_N +
        static_cast<int>(opus::thread_id_x()) * elems_per_thread;
    const int col_end = col_base + elems_per_thread;
    // Compile-time bound when TOPK>0 (launch guarantees TOPK == kargs.topk).
    const int topk_loop = (TOPK > 0) ? TOPK : kargs.topk;
    const int route_row_base = (TOPK > 0) ? token * TOPK : token * kargs.topk;

    // MXFP8 route_out: read fp8 e4m3 + per-8col e8m0 scale, dequant, sum over topk.
    // 8-col groups: one uint64 fp8 load + one e8m0 scale byte per group (scale is
    // per-8col so no redundant reads), 4 cvt + 8 FMA. ALU-bound -> minimal ops.
    if constexpr(ROUTE_FP8)
    {
        static_assert(elems_per_thread % 8 == 0);
        if(col_end > kargs.model_dim)
            return;

        const OpusMoeStage2RouteReduceMxFp8View route_view{
            kargs.route_out, kargs.route_out_row_bytes, kargs.model_dim};
        const OpusMoeStage2RouteReduceBf16OutView out_view{
            kargs.out_bf16, kargs.stride_o_t};
        // Interleave each 8-column read/sum/write group to keep loads wide while
        // overlapping the final bf16 writes.
        constexpr int NG = elems_per_thread / 8;
#pragma unroll
        for(int group_idx = 0; group_idx < NG; ++group_idx)
        {
            const int col = col_base + group_idx * 8;
            opus::fp32x4_t acc_lo{0.0f, 0.0f, 0.0f, 0.0f};
            opus::fp32x4_t acc_hi{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for(int slot = 0; slot < topk_loop; ++slot)
            {
                const auto mx_group = route_view.load_group(route_row_base + slot, col);
                acc_lo += opus_moe_stage2_route_reduce_dequant_fp8x4(
                    static_cast<uint32_t>(mx_group.data), mx_group.scale);
                acc_hi += opus_moe_stage2_route_reduce_dequant_fp8x4(
                    static_cast<uint32_t>(mx_group.data >> 32), mx_group.scale);
            }
            const uint32_t p01 = opus_moe_gfx950_cvt_pk_bf16_f32(acc_lo[0], acc_lo[1]);
            const uint32_t p23 = opus_moe_gfx950_cvt_pk_bf16_f32(acc_lo[2], acc_lo[3]);
            const uint32_t p45 = opus_moe_gfx950_cvt_pk_bf16_f32(acc_hi[0], acc_hi[1]);
            const uint32_t p67 = opus_moe_gfx950_cvt_pk_bf16_f32(acc_hi[2], acc_hi[3]);
            out_view.store_x4(
                token, col, static_cast<uint64_t>(p01) | (static_cast<uint64_t>(p23) << 32));
            out_view.store_x4(
                token, col + 4, static_cast<uint64_t>(p45) | (static_cast<uint64_t>(p67) << 32));
        }
        return;
    }
    else
    {
        static_assert(elems_per_thread % 4 == 0);
        constexpr int groups_per_thread = elems_per_thread / 4;
        const OpusMoeStage2RouteReduceBf16RouteView route_view{
            reinterpret_cast<const hip_bfloat16*>(kargs.route_out), kargs.stride_route_out_t};
        const OpusMoeStage2RouteReduceBf16OutView out_view{
            kargs.out_bf16, kargs.stride_o_t};
        if(col_end <= kargs.model_dim)
        {
            float acc[elems_per_thread];
#pragma unroll
            for(int j = 0; j < elems_per_thread; ++j)
            {
                acc[j] = 0.0f;
            }

#pragma unroll
            for(int slot = 0; slot < topk_loop; ++slot)
            {
                const int route_row = route_row_base + slot;
#pragma unroll
                for(int group = 0; group < groups_per_thread; ++group)
                {
                    const int col = col_base + group * 4;
                    const uint64_t packed = route_view.load_x4(route_row, col);
                    opus_moe_stage2_route_reduce_accum_bf16x4(acc, group * 4, packed);
                }
            }

#pragma unroll
            for(int group = 0; group < groups_per_thread; ++group)
            {
                const int col = col_base + group * 4;
                const uint64_t packed_out =
                    opus_moe_stage2_route_reduce_pack_bf16x4(acc, group * 4);
                out_view.store_x4(token, col, packed_out);
            }
            return;
        }

        if(col_base >= kargs.model_dim)
            return;

        constexpr int max_scalar_tail = 3;
        const int valid_elems = kargs.model_dim - col_base;
        const int valid_groups4 = valid_elems / 4;
        const int scalar_begin = valid_groups4 * 4;
        const int scalar_tail = valid_elems - scalar_begin;

        float acc[elems_per_thread];
#pragma unroll
        for(int group = 0; group < groups_per_thread; ++group)
        {
            if(group < valid_groups4)
            {
                const int base = group * 4;
                acc[base + 0] = 0.0f;
                acc[base + 1] = 0.0f;
                acc[base + 2] = 0.0f;
                acc[base + 3] = 0.0f;
            }
        }

        if(scalar_tail == 0)
        {
#pragma unroll
            for(int slot = 0; slot < topk_loop; ++slot)
            {
                const int route_row = route_row_base + slot;
#pragma unroll
                for(int group = 0; group < groups_per_thread; ++group)
                {
                    if(group < valid_groups4)
                    {
                        const int col = col_base + group * 4;
                        const uint64_t packed = route_view.load_x4(route_row, col);
                        opus_moe_stage2_route_reduce_accum_bf16x4(acc, group * 4, packed);
                    }
                }
            }

#pragma unroll
            for(int group = 0; group < groups_per_thread; ++group)
            {
                if(group < valid_groups4)
                {
                    const int col = col_base + group * 4;
                    out_view.store_x4(
                        token, col, opus_moe_stage2_route_reduce_pack_bf16x4(acc, group * 4));
                }
            }
            return;
        }

#pragma unroll
        for(int tail = 0; tail < max_scalar_tail; ++tail)
        {
            const int j = scalar_begin + tail;
            if(tail < scalar_tail)
            {
                acc[j] = 0.0f;
            }
        }

#pragma unroll
        for(int slot = 0; slot < topk_loop; ++slot)
        {
            const int route_row = route_row_base + slot;
#pragma unroll
            for(int group = 0; group < groups_per_thread; ++group)
            {
                if(group < valid_groups4)
                {
                    const int col = col_base + group * 4;
                    const uint64_t packed = route_view.load_x4(route_row, col);
                    opus_moe_stage2_route_reduce_accum_bf16x4(acc, group * 4, packed);
                }
            }
#pragma unroll
            for(int tail = 0; tail < max_scalar_tail; ++tail)
            {
                const int j = scalar_begin + tail;
                if(tail < scalar_tail)
                {
                    const int col = col_base + j;
                    const hip_bfloat16 value = route_view.load(route_row, col);
                    acc[j] += static_cast<float>(value);
                }
            }
        }

#pragma unroll
        for(int group = 0; group < groups_per_thread; ++group)
        {
            if(group < valid_groups4)
            {
                const int col = col_base + group * 4;
                out_view.store_x4(
                    token, col, opus_moe_stage2_route_reduce_pack_bf16x4(acc, group * 4));
            }
        }
#pragma unroll
        for(int tail = 0; tail < max_scalar_tail; ++tail)
        {
            const int j = scalar_begin + tail;
            if(tail < scalar_tail)
            {
                const int col = col_base + j;
                out_view.store(token, col, opus_moe_gfx950_cvt_bf16_f32(acc[j]));
            }
        }
    }
#endif
#endif
}
