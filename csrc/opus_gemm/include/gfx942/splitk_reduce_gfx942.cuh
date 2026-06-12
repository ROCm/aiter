// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Split-K reduce kernels (gfx942): sum workspace partials across split-K and write C[B, M, N].
#pragma once

#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a16w16.cuh"  // opus_splitk_ws_handle
#include <cstdint>

template<int VEC_ = 16, int BLOCK_ = 64, typename D_WS = float,
         typename D_OUT = __bf16, bool HAS_BIAS_ = false,
         typename D_BIAS_ = D_OUT, bool HAS_OOB_ = true>
__device__ __forceinline__ void splitk_reduce_kernel_fallback_body(
    const opus_splitk_ws_handle* __restrict__ ws_handle,
    D_OUT*       __restrict__ c_out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    const D_BIAS_* __restrict__ bias,
    int bias_stride_batch)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx942__)
    constexpr int VEC   = VEC_;
    constexpr int BLOCK = BLOCK_;
    constexpr bool HAS_BIAS = HAS_BIAS_;
    constexpr bool HAS_OOB = HAS_OOB_;
    using D_BIAS = D_BIAS_;
    static_assert(sizeof(D_WS) == 2 || sizeof(D_WS) == 4,
                  "splitk_reduce fallback supports fp32/bf16 workspace");

    constexpr int STEP = 16 / sizeof(D_OUT);
    static_assert(STEP * sizeof(D_OUT) == 16,
                  "D_OUT must divide a 128-bit store boundary cleanly "
                  "(supported sizes: 2B / 4B; e.g. __bf16, float)");
    static_assert(VEC % STEP == 0,
                  "VEC must be a multiple of STEP so the fast path tiles "
                  "into whole dwordx4 stores");
    static_assert(!HAS_BIAS || sizeof(D_BIAS) == 2 || sizeof(D_BIAS) == 4,
                  "splitk_reduce HAS_BIAS path supports only 2B or 4B D_BIAS "
                  "(bf16 / fp32). Other widths require half-extract changes.");

    const int bm_id  = int(opus::block_id_y());
    const int nblk   = int(opus::block_id_x());
    const int tid    = int(opus::thread_id_x());
    const int n_base = (nblk * BLOCK + tid) * VEC;

    const int b = bm_id / M;
    const int m = bm_id - b * M;
    const D_WS* workspace = reinterpret_cast<const D_WS*>(ws_handle->ptr);

    opus::vector_t<float, VEC> bias_fp32;
    if constexpr (HAS_BIAS) {
        #pragma unroll
        for (int t = 0; t < VEC; ++t) bias_fp32[t] = 0.0f;
        const D_BIAS* bias_base_ptr = bias + b * bias_stride_batch;
        auto g_bias = opus::make_gmem(bias_base_ptr,
                        (unsigned int)((bias_stride_batch ? bias_stride_batch : N)
                                       * sizeof(D_BIAS)));
        #pragma unroll
        for (int g = 0; g < VEC / 4; ++g) {
            auto bv4 = g_bias.template load<4>(n_base + g * 4);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                bias_fp32[g * 4 + j] = static_cast<float>(bv4[j]);
        }
    }

    const int  ws_row_base  = b * padded_M * padded_N + m * padded_N + n_base;
    const long split_stride = (long)batch * padded_M * padded_N;

    auto g_ws = opus::make_gmem(workspace,
                                (unsigned int)(split_stride * split_k * sizeof(D_WS)));

    opus::vector_t<float, VEC> acc;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) acc[t] = 0.0f;

    for (int s = 0; s < split_k; ++s) {
        int ws_idx = ws_row_base + (int)(s * split_stride);
        if constexpr (sizeof(D_WS) == 2 && VEC % 8 == 0) {
            #pragma unroll
            for (int g = 0; g < VEC / 8; ++g) {
                auto v8 = g_ws.template load<8>(ws_idx + g * 8);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[g * 8 + j] += static_cast<float>(v8[j]);
                }
            }
        } else {
            static_assert(VEC % 4 == 0,
                          "splitk_reduce fallback fp32 workspace needs VEC % 4 == 0");
            #pragma unroll
            for (int g = 0; g < VEC / 4; ++g) {
                auto v4 = g_ws.template load<4>(ws_idx + g * 4);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[g * 4 + j] += static_cast<float>(v4[j]);
                }
            }
        }
    }

    if constexpr (HAS_BIAS) {
        #pragma unroll
        for (int t = 0; t < VEC; ++t) acc[t] += bias_fp32[t];
    }

    opus::vector_t<D_OUT, VEC> out;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) out[t] = static_cast<D_OUT>(acc[t]);

    auto g_c = opus::make_gmem(c_out,
                               (unsigned int)((size_t)batch * M * N * sizeof(D_OUT)));
    const int c_idx = b * M * N + m * N + n_base;

    using opus::slice;
    using opus::number;
#define OPUS_REDUCE_ST8(OFF) g_c.template store<8>(slice(out, number<OFF>{}, number<OFF+8>{}), c_idx + (OFF))
#define OPUS_REDUCE_ST4(OFF) g_c.template store<4>(slice(out, number<OFF>{}, number<OFF+4>{}), c_idx + (OFF))
#define OPUS_REDUCE_ST2(OFF) g_c.template store<2>(slice(out, number<OFF>{}, number<OFF+2>{}), c_idx + (OFF))
#define OPUS_REDUCE_ST1(OFF) g_c.template store<1>(out[OFF], c_idx + (OFF))

    if constexpr (!HAS_OOB) {
        if (n_base + VEC <= N) {
            opus::static_for<VEC / STEP>([&](auto g_c_idx) {
                constexpr int g = decltype(g_c_idx)::value;
                g_c.template store<STEP>(
                    slice(out, number<g * STEP>{}, number<(g + 1) * STEP>{}),
                    c_idx + g * STEP);
            });
        }
    } else {
        if (n_base + VEC <= N) {
            opus::static_for<VEC / STEP>([&](auto g_c_idx) {
                constexpr int g = decltype(g_c_idx)::value;
                g_c.template store<STEP>(
                    slice(out, number<g * STEP>{}, number<(g + 1) * STEP>{}),
                    c_idx + g * STEP);
            });
        } else if (n_base < N) {
            static_assert(VEC == 16, "reduce tail switch assumes VEC=16");
            const int valid = N - n_base;
            if constexpr (sizeof(D_OUT) == 2) {
                switch (valid) {
                    case  1: OPUS_REDUCE_ST1( 0); break;
                    case  2: OPUS_REDUCE_ST2( 0); break;
                    case  3: OPUS_REDUCE_ST2( 0); OPUS_REDUCE_ST1( 2); break;
                    case  4: OPUS_REDUCE_ST4( 0); break;
                    case  5: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST1( 4); break;
                    case  6: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST2( 4); break;
                    case  7: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST2( 4); OPUS_REDUCE_ST1( 6); break;
                    case  8: OPUS_REDUCE_ST8( 0); break;
                    case  9: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST1( 8); break;
                    case 10: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST2( 8); break;
                    case 11: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST2( 8); OPUS_REDUCE_ST1(10); break;
                    case 12: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST4( 8); break;
                    case 13: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST1(12); break;
                    case 14: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST2(12); break;
                    case 15: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST2(12); OPUS_REDUCE_ST1(14); break;
                }
            } else {
                switch (valid) {
                    case  1: OPUS_REDUCE_ST1( 0); break;
                    case  2: OPUS_REDUCE_ST2( 0); break;
                    case  3: OPUS_REDUCE_ST2( 0); OPUS_REDUCE_ST1( 2); break;
                    case  4: OPUS_REDUCE_ST4( 0); break;
                    case  5: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST1( 4); break;
                    case  6: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST2( 4); break;
                    case  7: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST2( 4); OPUS_REDUCE_ST1( 6); break;
                    case  8: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); break;
                    case  9: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST1( 8); break;
                    case 10: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST2( 8); break;
                    case 11: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST2( 8); OPUS_REDUCE_ST1(10); break;
                    case 12: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST4( 8); break;
                    case 13: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST1(12); break;
                    case 14: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST2(12); break;
                    case 15: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST2(12); OPUS_REDUCE_ST1(14); break;
                }
            }
        }
    }
#undef OPUS_REDUCE_ST8
#undef OPUS_REDUCE_ST4
#undef OPUS_REDUCE_ST2
#undef OPUS_REDUCE_ST1
#else
    // Non-gfx942 device pass: empty stub.
#endif  // __gfx942__
#endif  // __HIP_DEVICE_COMPILE__
}

template<int VEC_ = 16, int BLOCK_ = 64, typename D_OUT = __bf16,
         bool HAS_BIAS_ = false, typename D_BIAS_ = D_OUT,
         bool HAS_OOB_ = true>
__global__ void splitk_reduce_kernel_fallback(
    const opus_splitk_ws_handle* __restrict__ ws_handle,
    D_OUT*       __restrict__ c_out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    const D_BIAS_* __restrict__ bias,
    int bias_stride_batch)
{
    splitk_reduce_kernel_fallback_body<VEC_, BLOCK_, float, D_OUT, HAS_BIAS_, D_BIAS_, HAS_OOB_>(
        ws_handle, c_out, split_k, M, N, batch, padded_M, padded_N, bias, bias_stride_batch);
}

template<int VEC_ = 16, int BLOCK_ = 64, typename D_OUT = __bf16,
         bool HAS_BIAS_ = false, typename D_BIAS_ = D_OUT,
         bool HAS_OOB_ = true>
__global__ void splitk_reduce_kernel_bf16ws_fallback(
    const opus_splitk_ws_handle* __restrict__ ws_handle,
    D_OUT*       __restrict__ c_out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    const D_BIAS_* __restrict__ bias,
    int bias_stride_batch)
{
    splitk_reduce_kernel_fallback_body<VEC_, BLOCK_, __bf16, D_OUT, HAS_BIAS_, D_BIAS_, HAS_OOB_>(
        ws_handle, c_out, split_k, M, N, batch, padded_M, padded_N, bias, bias_stride_batch);
}

// Exact-N row-block fast path: static split_k unroll, no N tail/OOB.
template<int SPLIT_K, int N_VEC, int ROWS_PER_BLOCK, int VEC_ = 8,
         typename D_WS = float, typename D_OUT = __bf16,
         bool HAS_BIAS_ = false, typename D_BIAS_ = D_OUT>
__global__ void splitk_reduce_kernel_exact_n_rowblock(
    const opus_splitk_ws_handle* __restrict__ ws_handle,
    D_OUT*       __restrict__ c_out,
    int M, int N, int batch,
    int padded_M, int padded_N,
    const D_BIAS_* __restrict__ bias,
    int bias_stride_batch)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx942__)
    constexpr int VEC = VEC_;
    constexpr int BLOCK = N_VEC * ROWS_PER_BLOCK;
    constexpr bool HAS_BIAS = HAS_BIAS_;
    using D_BIAS = D_BIAS_;
    static_assert(sizeof(D_WS) == 2 || sizeof(D_WS) == 4,
                  "exact-N row-block reduce supports fp32/bf16 workspace");

    constexpr int STEP = 16 / sizeof(D_OUT);
    static_assert(STEP * sizeof(D_OUT) == 16);
    static_assert(VEC % STEP == 0);

    const int bm_blk = int(opus::block_id_y());  // 0 .. (M / ROWS_PER_BLOCK - 1)
    const int b      = int(opus::block_id_z());
    const int tid    = int(opus::thread_id_x());

    // Thread layout: tid = (row_off, n_vec) with row_off = tid / N_VEC, n_vec = tid % N_VEC
    const int row_off = tid / N_VEC;
    const int n_vec   = tid - row_off * N_VEC;
    const int m       = bm_blk * ROWS_PER_BLOCK + row_off;
    const int n_base  = n_vec * VEC;
    const D_WS* workspace = reinterpret_cast<const D_WS*>(ws_handle->ptr);

    opus::vector_t<float, VEC> bias_fp32;
    if constexpr (HAS_BIAS) {
        #pragma unroll
        for (int t = 0; t < VEC; ++t) bias_fp32[t] = 0.0f;
        const D_BIAS* bias_base_ptr = bias + b * bias_stride_batch;
        auto g_bias = opus::make_gmem(bias_base_ptr,
                        (unsigned int)((bias_stride_batch ? bias_stride_batch : N)
                                       * sizeof(D_BIAS)));
        #pragma unroll
        for (int g = 0; g < VEC / 4; ++g) {
            auto bv4 = g_bias.template load<4>(n_base + g * 4);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                bias_fp32[g * 4 + j] = static_cast<float>(bv4[j]);
        }
    }

    const int  ws_row_base  = b * padded_M * padded_N + m * padded_N + n_base;
    const long split_stride = (long)batch * padded_M * padded_N;

    auto g_ws = opus::make_gmem(workspace,
                                (unsigned int)(split_stride * SPLIT_K * sizeof(D_WS)));

    opus::vector_t<float, VEC> acc;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) acc[t] = 0.0f;

    // Direct accumulation keeps the fixed split order without staging partials.
    #pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
        int ws_idx = ws_row_base + (int)(s * split_stride);
        if constexpr (sizeof(D_WS) == 2 && VEC % 8 == 0) {
            #pragma unroll
            for (int g = 0; g < VEC / 8; ++g) {
                auto v8 = g_ws.template load<8>(ws_idx + g * 8);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[g * 8 + j] += static_cast<float>(v8[j]);
                }
            }
        } else {
            static_assert(VEC % 4 == 0,
                          "exact-N row-block reduce fp32 path needs VEC % 4 == 0");
            #pragma unroll
            for (int g = 0; g < VEC / 4; ++g) {
                auto v4 = g_ws.template load<4>(ws_idx + g * 4);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[g * 4 + j] += static_cast<float>(v4[j]);
                }
            }
        }
    }

    if constexpr (HAS_BIAS) {
        #pragma unroll
        for (int t = 0; t < VEC; ++t) acc[t] += bias_fp32[t];
    }

    opus::vector_t<D_OUT, VEC> out;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) out[t] = static_cast<D_OUT>(acc[t]);

    auto g_c = opus::make_gmem(c_out,
                               (unsigned int)((size_t)batch * M * N * sizeof(D_OUT)));
    const int c_idx = b * M * N + m * N + n_base;

    using opus::slice;
    using opus::number;
    opus::static_for<VEC / STEP>([&](auto g_c_idx) {
        constexpr int g = decltype(g_c_idx)::value;
        g_c.template store<STEP>(
            slice(out, number<g * STEP>{}, number<(g + 1) * STEP>{}),
            c_idx + g * STEP);
    });
#endif  // __gfx942__
#endif  // __HIP_DEVICE_COMPILE__
}
