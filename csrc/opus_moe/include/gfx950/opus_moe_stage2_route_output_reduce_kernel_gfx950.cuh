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
    const hip_bfloat16* __restrict__ route_out_bf16 =
        reinterpret_cast<const hip_bfloat16*>(kargs.route_out);

    // MXFP8 route_out: read fp8 e4m3 + per-8col e8m0 scale, dequant, sum over topk.
    // 8-col groups: one uint64 fp8 load + one e8m0 scale byte per group (scale is
    // per-8col so no redundant reads), 4 cvt + 8 FMA. ALU-bound -> minimal ops.
    if constexpr(ROUTE_FP8)
    {
        static_assert(elems_per_thread % 8 == 0);
        if(col_end > kargs.model_dim)
            return;

        using f32x2 = float __attribute__((ext_vector_type(2)));
        const uint8_t* __restrict__ ro8 = kargs.route_out;
        const int64_t rstride = kargs.route_out_row_bytes;
        const int scale_off = kargs.model_dim;
        // Interleave each 8-column read/sum/write group to keep loads wide while
        // overlapping the final bf16 writes.
        constexpr int NG = elems_per_thread / 8;
#pragma unroll
        for(int group = 0; group < NG; ++group)
        {
            const int col = col_base + group * 8;
            f32x2 acc[4] = {f32x2{0.0f, 0.0f}, f32x2{0.0f, 0.0f},
                            f32x2{0.0f, 0.0f}, f32x2{0.0f, 0.0f}};
#pragma unroll
            for(int slot = 0; slot < topk_loop; ++slot)
            {
                const uint8_t* rp =
                    ro8 + static_cast<int64_t>(route_row_base + slot) * rstride;
                const uint64_t f8 = *reinterpret_cast<const uint64_t*>(rp + col);
                const float sf = __builtin_bit_cast(
                    float, static_cast<uint32_t>(rp[scale_off + (col >> 3)]) << 23);
                const f32x2 s2 = f32x2{sf, sf};
                const uint32_t lo32 = static_cast<uint32_t>(f8);
                const uint32_t hi32 = static_cast<uint32_t>(f8 >> 32);
                acc[0] += __builtin_amdgcn_cvt_pk_f32_fp8(lo32, false) * s2;
                acc[1] += __builtin_amdgcn_cvt_pk_f32_fp8(lo32, true) * s2;
                acc[2] += __builtin_amdgcn_cvt_pk_f32_fp8(hi32, false) * s2;
                acc[3] += __builtin_amdgcn_cvt_pk_f32_fp8(hi32, true) * s2;
            }
            const uint32_t p01 = opus_moe_gfx950_cvt_pk_bf16_f32(acc[0][0], acc[0][1]);
            const uint32_t p23 = opus_moe_gfx950_cvt_pk_bf16_f32(acc[1][0], acc[1][1]);
            const uint32_t p45 = opus_moe_gfx950_cvt_pk_bf16_f32(acc[2][0], acc[2][1]);
            const uint32_t p67 = opus_moe_gfx950_cvt_pk_bf16_f32(acc[3][0], acc[3][1]);
            hip_bfloat16* op = kargs.out_bf16 +
                               static_cast<int64_t>(token) * kargs.stride_o_t + col;
            reinterpret_cast<uint64_t*>(op)[0] =
                static_cast<uint64_t>(p01) | (static_cast<uint64_t>(p23) << 32);
            reinterpret_cast<uint64_t*>(op)[1] =
                static_cast<uint64_t>(p45) | (static_cast<uint64_t>(p67) << 32);
        }
        return;
    }
    else
    {
        static_assert(elems_per_thread % 4 == 0);
        constexpr int groups_per_thread = elems_per_thread / 4;
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
                    const uint64_t packed =
                        *reinterpret_cast<const uint64_t*>(
                            route_out_bf16 +
                            static_cast<int64_t>(route_row) * kargs.stride_route_out_t + col);
                    opus_moe_stage2_route_reduce_accum_bf16x4(acc, group * 4, packed);
                }
            }

#pragma unroll
            for(int group = 0; group < groups_per_thread; ++group)
            {
                const int col = col_base + group * 4;
                const uint64_t packed_out =
                    opus_moe_stage2_route_reduce_pack_bf16x4(acc, group * 4);
                *reinterpret_cast<uint64_t*>(kargs.out_bf16 +
                                             static_cast<int64_t>(token) *
                                                 kargs.stride_o_t +
                                             col) = packed_out;
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
                        const uint64_t packed =
                            *reinterpret_cast<const uint64_t*>(
                                route_out_bf16 +
                                static_cast<int64_t>(route_row) * kargs.stride_route_out_t + col);
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
                    *reinterpret_cast<uint64_t*>(kargs.out_bf16 +
                                                 static_cast<int64_t>(token) *
                                                     kargs.stride_o_t +
                                                 col) =
                        opus_moe_stage2_route_reduce_pack_bf16x4(acc, group * 4);
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
                    const uint64_t packed =
                        *reinterpret_cast<const uint64_t*>(
                            route_out_bf16 +
                            static_cast<int64_t>(route_row) * kargs.stride_route_out_t + col);
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
                    const hip_bfloat16 value =
                        route_out_bf16[static_cast<int64_t>(route_row) *
                                            kargs.stride_route_out_t +
                                        col];
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
                *reinterpret_cast<uint64_t*>(kargs.out_bf16 +
                                             static_cast<int64_t>(token) * kargs.stride_o_t +
                                             col) =
                    opus_moe_stage2_route_reduce_pack_bf16x4(acc, group * 4);
            }
        }
#pragma unroll
        for(int tail = 0; tail < max_scalar_tail; ++tail)
        {
            const int j = scalar_begin + tail;
            if(tail < scalar_tail)
            {
                const int col = col_base + j;
                kargs.out_bf16[static_cast<int64_t>(token) * kargs.stride_o_t + col] =
                    opus_moe_gfx950_cvt_bf16_f32(acc[j]);
            }
        }
    }
#endif
#endif
}
