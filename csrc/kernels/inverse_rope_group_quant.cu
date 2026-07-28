// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_dispatch.h"
#include "aiter_stream.h"
#include "inverse_rope_group_quant.h"
#include "mx_quant_utils.h"
#include "opus/opus.hpp"

#include <bit>
#include <cmath>
#include <type_traits>

#define CHECK_CONTIGUOUS(x) AITER_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace aiter {

static constexpr float kAbsmaxFloor = 1e-8f;

static constexpr MxDtype kHwFp8E4m3 =
#if defined(__gfx942__)
    MxDtype::FP8_E4M3_FNUZ;
#else
    MxDtype::FP8_E4M3;
#endif

template <int N>
using ic = std::integral_constant<int, N>;

// Cross-lane float max over THREADS_PER_GROUP adjacent lanes.
// gfx950+: __builtin_amdgcn_update_dpp keeps the DPP visible to the scheduler
// so the 2-wait-state hazard is honoured; permlane16/32_swap for 32/64 lanes.
// gfx942: falls back to wave_reduce (rocprim warp_move_dpp + row_bcast).
#if !defined(__gfx942__)
template <int dpp_ctrl>
__device__ __forceinline__ float dpp_max_step(float v)
{
    const int shuffled = __builtin_amdgcn_update_dpp(
        0, __builtin_bit_cast(int, v), dpp_ctrl, 0xf, 0xf, true);
    return fmaxf(v, __builtin_bit_cast(float, shuffled));
}
#endif

template <int thread_num>
__device__ __forceinline__ float group_reduce_max(float v)
{
    static_assert(thread_num >= 1 && thread_num <= 64 &&
                      (thread_num & (thread_num - 1)) == 0,
                  "thread_num must be a power of two in [1,64]");
#if defined(__gfx942__)
    auto fmax_op = [](float a, float b) { return fmaxf(a, b); };
    return wave_reduce<float, decltype(fmax_op), thread_num>(v, fmax_op);
#else
    if constexpr(thread_num >= 2)
    {
        v = dpp_max_step<0xb1>(v);
    }
    if constexpr(thread_num >= 4)
    {
        v = dpp_max_step<0x4e>(v);
    }
    if constexpr(thread_num >= 8)
    {
        v = dpp_max_step<0x141>(v);
    }
    if constexpr(thread_num >= 16)
    {
        v = dpp_max_step<0x140>(v);
    }
    if constexpr(thread_num >= 32)
    {
        const opus::vector_t<opus::u32_t, 2> sw = __builtin_amdgcn_permlane16_swap(
            std::bit_cast<opus::u32_t>(v), std::bit_cast<opus::u32_t>(v), false, true);
        v = fmaxf(std::bit_cast<float>(sw.x), std::bit_cast<float>(sw.y));
    }
    if constexpr(thread_num == 64)
    {
        const opus::vector_t<opus::u32_t, 2> sw = __builtin_amdgcn_permlane32_swap(
            std::bit_cast<opus::u32_t>(v), std::bit_cast<opus::u32_t>(v), false, true);
        v = fmaxf(std::bit_cast<float>(sw.x), std::bit_cast<float>(sw.y));
    }
    return v;
#endif
}

template <typename scalar_t,
          int HEAD_DIM,
          int RD,
          int GROUP_SIZE,
          int THREAD_DATA_SIZE,
          int BLOCK_M,
          int K_PER_BLOCK>
__global__ void inverse_rope_group_quant_kernel(
    const scalar_t* __restrict__ o,
    opus::fp8_t* __restrict__ x_fp8,
    uint8_t* __restrict__ x_scale,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ cos_cache,
    const scalar_t* __restrict__ sin_cache,
    int S,
    int H,
    int G,
    int D,
    int scale_n,
    int64_t scale_stride_s,
    int64_t scale_stride_g,
    int64_t scale_stride_k,
    int max_position)
{
    constexpr int THREADS_PER_GROUP = GROUP_SIZE / THREAD_DATA_SIZE;
    constexpr int BLOCK_SIZE = BLOCK_M * THREADS_PER_GROUP;
    static_assert(HEAD_DIM > 0 && RD > 0 && RD <= HEAD_DIM && (RD % 2) == 0);
    static_assert(GROUP_SIZE == 32 || GROUP_SIZE == 64 || GROUP_SIZE == 128);
    static_assert(HEAD_DIM % GROUP_SIZE == 0);
    static_assert(THREADS_PER_GROUP >= 1 && THREADS_PER_GROUP <= 64);

    constexpr int GROUPS_PER_HEAD = HEAD_DIM / GROUP_SIZE;
    constexpr int ROPE_START = HEAD_DIM - RD;

    const int tid = threadIdx.x;
    const int row_in_tile = tid / THREADS_PER_GROUP;
    const int lane_in_group = tid - row_in_tile * THREADS_PER_GROUP;
    const int row = static_cast<int>(blockIdx.y) * BLOCK_M + row_in_tile;
    const int k_group_base = static_cast<int>(blockIdx.x) * K_PER_BLOCK;
    if(row >= S * G || k_group_base >= scale_n)
    {
        return;
    }

    const int s = row / G;
    const int g = row - s * G;
    const int group_elem_base = lane_in_group * THREAD_DATA_SIZE;
    const int d_base0 = k_group_base * GROUP_SIZE + group_elem_base;

    // --- Load input ---
    using vec_i = opus::vector_t<scalar_t, THREAD_DATA_SIZE>;
    auto input_buffer = opus::make_gmem<scalar_t>(
        o, static_cast<int64_t>(S) * H * HEAD_DIM * sizeof(scalar_t));
    const int64_t input_offset0 =
        static_cast<int64_t>(s) * H * HEAD_DIM + static_cast<int64_t>(g) * D + d_base0;
    constexpr int in_chunk_bytes =
        (THREAD_DATA_SIZE * sizeof(scalar_t)) % 16 == 0 ? 16 :
        ((THREAD_DATA_SIZE * sizeof(scalar_t)) % 8 == 0 ? 8 : 4);

    // Issue all loads before consuming any -- K_PER_BLOCK independent loads give
    // the wave that many requests in flight for latency hiding.
    vec_i in_vec[K_PER_BLOCK];
#pragma unroll
    for(int k = 0; k < K_PER_BLOCK; ++k)
    {
        in_vec[k] = load_vector_nbytes<scalar_t, THREAD_DATA_SIZE, in_chunk_bytes>(
            input_buffer, input_offset0 + static_cast<int64_t>(k) * GROUP_SIZE);
    }

    // --- Determine if any group in this tile overlaps the rope tail ---
    bool any_rope = false;
#pragma unroll
    for(int k = 0; k < K_PER_BLOCK; ++k)
    {
        const int bhs = ((k_group_base + k) % GROUPS_PER_HEAD) * GROUP_SIZE;
        any_rope = any_rope || (bhs + GROUP_SIZE > ROPE_START);
    }
    int64_t pos = 0;
    if(any_rope)
    {
        pos = positions[s];
        if(pos < 0) pos = 0;
        if(max_position > 0 && pos >= max_position) pos = max_position - 1;
    }

    // --- Output buffer ---
    auto out_buffer = opus::make_gmem<opus::fp8_t>(
        x_fp8, static_cast<int64_t>(S) * G * D * sizeof(opus::fp8_t));
    const int64_t scale_row_idx = static_cast<int64_t>(s) * scale_stride_s +
                                  static_cast<int64_t>(g) * scale_stride_g;

    // --- Per-group: rope -> amax -> scale -> quantize -> store ---
#pragma unroll
    for(int k = 0; k < K_PER_BLOCK; ++k)
    {
        const int k_group = k_group_base + k;
        const int d_base = d_base0 + k * GROUP_SIZE;

        float vals[THREAD_DATA_SIZE];
#pragma unroll
        for(int i = 0; i < THREAD_DATA_SIZE; ++i)
        {
            vals[i] = static_cast<float>(in_vec[k][i]);
        }

        // --- Inverse RoPE on the rope tail ---
        const int block_head_start = (k_group % GROUPS_PER_HEAD) * GROUP_SIZE;
        const bool block_has_rope = block_head_start + GROUP_SIZE > ROPE_START;

        if(block_has_rope)
        {
            float orig[THREAD_DATA_SIZE];
#pragma unroll
            for(int i = 0; i < THREAD_DATA_SIZE; ++i)
            {
                orig[i] = vals[i];
            }

            constexpr int NCOS = THREAD_DATA_SIZE / 2;
            const int local0 = block_head_start + group_elem_base - ROPE_START;
            bool vectorized = false;
            if constexpr(NCOS >= 1)
            {
                if(local0 >= 0)
                {
                    using vec_c = opus::vector_t<scalar_t, NCOS>;
                    const int64_t crow = pos * (RD / 2) + (local0 >> 1);
                    const vec_c cvec = *reinterpret_cast<const vec_c*>(cos_cache + crow);
                    const vec_c svec = *reinterpret_cast<const vec_c*>(sin_cache + crow);
#pragma unroll
                    for(int i = 0; i < THREAD_DATA_SIZE; ++i)
                    {
                        const float c = static_cast<float>(cvec[i >> 1]);
                        const float sn = static_cast<float>(svec[i >> 1]);
                        const float val = orig[i];
                        const float pair = orig[i ^ 1];
                        vals[i] = (i & 1) == 0 ? (val * c + pair * sn)
                                                : (val * c - pair * sn);
                    }
                    vectorized = true;
                }
            }
            if(!vectorized)
            {
#pragma unroll
                for(int i = 0; i < THREAD_DATA_SIZE; ++i)
                {
                    const int group_elem = group_elem_base + i;
                    const int hd = block_head_start + group_elem;
                    if(hd >= ROPE_START)
                    {
                        const int local = hd - ROPE_START;
                        const int cos_i = local >> 1;
                        const float c = static_cast<float>(
                            cos_cache[pos * (RD / 2) + cos_i]);
                        const float sn = static_cast<float>(
                            sin_cache[pos * (RD / 2) + cos_i]);
                        const float val = orig[i];
                        const float pair = orig[i ^ 1];
                        vals[i] = (local & 1) == 0 ? (val * c + pair * sn)
                                                   : (val * c - pair * sn);
                    }
                }
            }
        }

        // --- Group amax reduction ---
        float amax = kAbsmaxFloor;
#pragma unroll
        for(int i = 0; i < THREAD_DATA_SIZE; ++i)
        {
            amax = fmaxf(amax, fabsf(vals[i]));
        }
        if constexpr(THREADS_PER_GROUP > 1)
        {
            amax = group_reduce_max<THREADS_PER_GROUP>(amax);
        }

        // --- E8M0 block scale ---
        const E8m0BlockScale s8 =
            fp_f32_to_e8m0_block_scale<MxScaleRoundMode::RoundUp, kHwFp8E4m3>(amax);
        const float inv_scale = 1.0f / s8.dq_scale;

        if(lane_in_group == 0)
        {
            x_scale[scale_row_idx + static_cast<int64_t>(k_group) * scale_stride_k] =
                s8.byte;
        }

        // --- Quantize and store ---
        if constexpr(THREAD_DATA_SIZE < 4)
        {
#pragma unroll
            for(int i = 0; i < THREAD_DATA_SIZE; ++i)
            {
                x_fp8[static_cast<int64_t>(row) * D + d_base + i] =
                    opus::cast<opus::fp8_t>(vals[i] * inv_scale);
            }
        }
        else
        {
            opus::vector_t<float, THREAD_DATA_SIZE> vec_vals;
#pragma unroll
            for(int i = 0; i < THREAD_DATA_SIZE; ++i)
            {
                vec_vals[i] = vals[i];
            }
            store_vector<opus::fp8_t, float, THREAD_DATA_SIZE, 0, false,
                         WARP_SIZE, 1, opus::fp8_t>(
                out_buffer, vec_vals, static_cast<int64_t>(row) * D + d_base, inv_scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

void inverse_rope_group_quant(
    aiter_tensor_t& o,
    aiter_tensor_t& x_fp8,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& positions,
    aiter_tensor_t& cos_cache,
    aiter_tensor_t& sin_cache,
    int64_t num_groups,
    int64_t quant_group_size,
    bool transpose_scale)
{
    AITER_CHECK(o.dim() == 3, "o must be [S,H,head_dim]");
    AITER_CHECK(x_fp8.dim() == 3, "x_fp8 must be [S,G,D]");
    AITER_CHECK(x_scale.dim() == 3, "x_scale must be [S,G,scale_n]");
    AITER_CHECK(o.dtype() == AITER_DTYPE_bf16 || o.dtype() == AITER_DTYPE_fp16,
                "o must be bf16/fp16, got ", AiterDtype_to_str(o.dtype()));
    AITER_CHECK(x_fp8.dtype() == AITER_DTYPE_fp8, "x_fp8 must be fp8");
    AITER_CHECK(x_scale.dtype() == AITER_DTYPE_fp8_e8m0 ||
                    x_scale.dtype() == AITER_DTYPE_u8,
                "x_scale must be fp8_e8m0 or uint8");
    AITER_CHECK(positions.dtype() == AITER_DTYPE_i64, "positions must be int64");
    AITER_CHECK(cos_cache.dim() == 2 && sin_cache.dim() == 2,
                "cos_cache/sin_cache must be 2D [max_pos, rd/2]");
    AITER_CHECK(cos_cache.dtype() == o.dtype() && sin_cache.dtype() == o.dtype(),
                "cos/sin dtype must match o");
    CHECK_CONTIGUOUS(o);
    CHECK_CONTIGUOUS(x_fp8);
    CHECK_CONTIGUOUS(cos_cache);
    CHECK_CONTIGUOUS(sin_cache);

    const int S = static_cast<int>(o.size(0));
    const int H = static_cast<int>(o.size(1));
    const int head_dim = static_cast<int>(o.size(2));
    const int G = static_cast<int>(num_groups);
    const int rd = static_cast<int>(cos_cache.size(1) * 2);
    AITER_CHECK(sin_cache.size(0) == cos_cache.size(0) &&
                    sin_cache.size(1) == cos_cache.size(1),
                "sin_cache shape must match cos_cache");
    AITER_CHECK(G > 0 && (H * head_dim) % G == 0,
                "H*head_dim must be divisible by num_groups");
    const int D = (H * head_dim) / G;
    AITER_CHECK(x_fp8.size(0) == S && x_fp8.size(1) == G && x_fp8.size(2) == D,
                "x_fp8 shape mismatch");
    AITER_CHECK(quant_group_size == 32 || quant_group_size == 64 ||
                    quant_group_size == 128,
                "quant_group_size must be one of {32,64,128}");
    AITER_CHECK(D % quant_group_size == 0,
                "D must be divisible by quant_group_size");
    AITER_CHECK(head_dim % quant_group_size == 0,
                "head_dim must be divisible by quant_group_size");
    AITER_CHECK(head_dim == 512 && rd == 64,
                "template path supports HEAD_DIM=512, RD=64; got ",
                head_dim, ",", rd);
    const int scale_n = D / static_cast<int>(quant_group_size);
    AITER_CHECK(x_scale.size(0) == S && x_scale.size(1) == G &&
                    x_scale.size(2) >= scale_n,
                "x_scale shape mismatch");

    if(transpose_scale)
    {
        AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(2) == S,
                    "transpose_scale expects column-major x_scale [G, scale_n, S]; "
                    "got strides ", x_scale.stride(0), ",", x_scale.stride(1),
                    ",", x_scale.stride(2));
    }
    else
    {
        AITER_CHECK(x_scale.stride(2) == 1,
                    "row-major x_scale must have unit stride along scale_n; "
                    "got stride ", x_scale.stride(2));
    }
    AITER_CHECK(rd > 0 && rd <= head_dim && (rd % 2) == 0, "invalid rotary dim");
    AITER_CHECK(positions.size(0) >= S, "positions length must be >= S");

    HipDeviceGuard device_guard(o.device_id);
    const hipStream_t stream = getCurrentHIPStream();

    constexpr int BLOCK_M = 16;

    auto launch = [&](auto group_tag, auto tds_tag, auto kpb_tag)
    {
        constexpr int GS = decltype(group_tag)::value;
        constexpr int HEAD_DIM_T = 512;
        constexpr int RD_T = 64;
        constexpr int TDS = decltype(tds_tag)::value;
        constexpr int THREADS_PER_GROUP = GS / TDS;
        constexpr int KPB = decltype(kpb_tag)::value;
        constexpr int BLOCK_SIZE = BLOCK_M * THREADS_PER_GROUP;
        if constexpr(BLOCK_SIZE > 1024 || THREADS_PER_GROUP < 1)
        {
            AITER_CHECK(false, "invalid THREAD_DATA_SIZE/BLOCK_M combination");
        }
        else
        {
            const dim3 grid(scale_n / KPB, (S * G + BLOCK_M - 1) / BLOCK_M);
            const dim3 block(BLOCK_SIZE);
            AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
                o.dtype(), "inverse_rope_group_quant", [&]
            {
                using scalar_opus_t = typename hip2opus<scalar_t>::type;
                inverse_rope_group_quant_kernel<
                    scalar_opus_t, HEAD_DIM_T, RD_T, GS, TDS, BLOCK_M, KPB>
                    <<<grid, block, 0, stream>>>(
                        reinterpret_cast<const scalar_opus_t*>(o.data_ptr()),
                        reinterpret_cast<opus::fp8_t*>(x_fp8.data_ptr()),
                        reinterpret_cast<uint8_t*>(x_scale.data_ptr()),
                        reinterpret_cast<const int64_t*>(positions.data_ptr()),
                        reinterpret_cast<const scalar_opus_t*>(cos_cache.data_ptr()),
                        reinterpret_cast<const scalar_opus_t*>(sin_cache.data_ptr()),
                        S, H, G, D, scale_n,
                        x_scale.stride(0), x_scale.stride(1), x_scale.stride(2),
                        static_cast<int>(cos_cache.size(0)));
            });
        }
    };

    auto dispatch_kpb = [&](auto group_tag, auto tds_tag, int kpb)
    {
        switch(kpb)
        {
            case 2: launch(group_tag, tds_tag, ic<2>{}); break;
            case 4: launch(group_tag, tds_tag, ic<4>{}); break;
            default: launch(group_tag, tds_tag, ic<1>{}); break;
        }
    };

    auto dispatch = [&](auto group_tag)
    {
        // Dispatch tiers from TDS x KPB sweep on MI355X (gfx950):
        // Small S: few rows -> keep one group per block (KPB=1).
        // Large S: bandwidth bound -> multiple loads in flight (KPB=4).
        const int tds = (S <= 4) ? 2 : (S <= 128 ? 4 : 8);
        int kpb = (S <= 128) ? 1 : (S <= 512 ? 2 : 4);
        while(kpb > 1 && scale_n % kpb != 0)
        {
            kpb >>= 1;
        }
        switch(tds)
        {
            case 2: dispatch_kpb(group_tag, ic<2>{}, kpb); break;
            case 4: dispatch_kpb(group_tag, ic<4>{}, kpb); break;
            default: dispatch_kpb(group_tag, ic<8>{}, kpb); break;
        }
    };

    if(quant_group_size == 32)
    {
        dispatch(ic<32>{});
    }
    else if(quant_group_size == 64)
    {
        dispatch(ic<64>{});
    }
    else
    {
        dispatch(ic<128>{});
    }
}

} // namespace aiter
