// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 split-K reduce kernel: tile-agnostic; sums an fp32 workspace across
// the split-K axis, folds an optional per-N bias once, casts fp32 -> D_OUT,
// and writes C. The body mirrors gfx950/splitk_reduce_gfx950.cuh (same
// ws_handle ABI), but the kernel is given a DISTINCT name
// (splitk_reduce_kernel_gfx1250) so the explicit instantiations do NOT collide
// with gfx950's identically-signatured splitk_reduce_kernel<ws_handle*> in a
// multi-arch build (same mangled name + same ABI would be a duplicate symbol).
//
// The reduce path uses no WMMA -- on wave32 BLOCK=64 is simply 2 waves of
// vectorized fp32 loads / adds / casts, identical to the gfx950 body.
//
// Grid: (ceil(N, VEC * BLOCK), batch * M, 1); all launchers use VEC=16, BLOCK=64.
#pragma once

#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a16w16_gfx1250.cuh"  // opus_splitk_ws_handle
#include <cstdint>

template<int VEC_ = 16, int BLOCK_ = 64, typename D_OUT = __bf16,
         bool HAS_BIAS_ = false, typename D_BIAS_ = D_OUT,
         bool HAS_OOB_ = true>
__global__ void splitk_reduce_kernel_gfx1250(
    const opus_splitk_ws_handle* __restrict__ ws_handle,
    D_OUT*       __restrict__ c_out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    const D_BIAS_* __restrict__ bias = nullptr,
    int bias_stride_batch = 0)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx1250__)
    const float* __restrict__ workspace =
        reinterpret_cast<const float*>(ws_handle->ptr);
    constexpr int VEC   = VEC_;
    constexpr int BLOCK = BLOCK_;
    constexpr bool HAS_BIAS = HAS_BIAS_;
    constexpr bool HAS_OOB = HAS_OOB_;
    using D_BIAS = D_BIAS_;

    constexpr int STEP = 16 / sizeof(D_OUT);
    static_assert(STEP * sizeof(D_OUT) == 16,
                  "D_OUT must divide a 128-bit store boundary cleanly (2B / 4B)");
    static_assert(VEC % STEP == 0,
                  "VEC must be a multiple of STEP so the fast path tiles into whole dwordx4 stores");
    static_assert(!HAS_BIAS || sizeof(D_BIAS) == 2 || sizeof(D_BIAS) == 4,
                  "splitk_reduce HAS_BIAS path supports only 2B or 4B D_BIAS (bf16 / fp32)");

    const int bm_id  = int(opus::block_id_y());
    const int nblk   = int(opus::block_id_x());
    const int tid    = int(opus::thread_id_x());
    const int n_base = (nblk * BLOCK + tid) * VEC;

    const int b = bm_id / M;
    const int m = bm_id - b * M;

    opus::vector_t<float, VEC> bias_fp32;
    if constexpr (HAS_BIAS) {
        #pragma unroll
        for (int t = 0; t < VEC; ++t) bias_fp32[t] = 0.0f;
        const D_BIAS* bias_base_ptr = bias + b * bias_stride_batch;
        auto g_bias = opus::make_gmem(bias_base_ptr,
                        (unsigned int)((bias_stride_batch ? bias_stride_batch : N) * sizeof(D_BIAS)));
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
                                (unsigned int)(split_stride * split_k * sizeof(float)));

    opus::vector_t<float, VEC> acc;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) acc[t] = 0.0f;

    for (int s = 0; s < split_k; ++s) {
        int ws_idx = ws_row_base + (int)(s * split_stride);
        #pragma unroll
        for (int g = 0; g < VEC / 4; ++g) {
            auto v4 = g_ws.template load<4>(ws_idx + g * 4);
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[g * 4 + j] += v4[j];
        }
    }

    if constexpr (HAS_BIAS) {
        #pragma unroll
        for (int t = 0; t < VEC; ++t) acc[t] += bias_fp32[t];
    }

    opus::vector_t<D_OUT, VEC> out;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) out[t] = static_cast<D_OUT>(acc[t]);

    auto g_c = opus::make_gmem(c_out, (unsigned int)((size_t)batch * M * N * sizeof(D_OUT)));
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
                    slice(out, number<g * STEP>{}, number<(g + 1) * STEP>{}), c_idx + g * STEP);
            });
        }
    } else {
        if (n_base + VEC <= N) {
            opus::static_for<VEC / STEP>([&](auto g_c_idx) {
                constexpr int g = decltype(g_c_idx)::value;
                g_c.template store<STEP>(
                    slice(out, number<g * STEP>{}, number<(g + 1) * STEP>{}), c_idx + g * STEP);
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
    (void)ws_handle; (void)c_out; (void)split_k; (void)M; (void)N; (void)batch;
    (void)padded_M; (void)padded_N; (void)bias; (void)bias_stride_batch;
#endif  // __gfx1250__
#endif  // __HIP_DEVICE_COMPILE__
}
