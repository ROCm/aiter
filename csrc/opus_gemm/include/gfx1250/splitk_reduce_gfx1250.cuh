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
// The reduce path uses no WMMA -- on wave32 it is simply vectorized fp32
// loads / adds / casts.
//
// ── Coalesced store, no cross-lane shuffle (triton-aligned) ────────────────
// The launchers use VEC=8, so each lane owns exactly 8 bf16 = 16B = one
// buffer_store_dwordx4. Lane l writes C[.. + l*8 .. +7], i.e. byte offset
// l*16 at a 16B stride -> the 32 lanes of a wave write 32*16 = 512B FULLY
// CONTIGUOUS in ONE store instruction: two 256B cache lines, each filled by
// 16 consecutive lanes (100% write-transaction efficiency). This drops the
// old VEC=16 layout, whose store<8> wrote 16B at a 32B stride and half-filled
// every 256B line, WITHOUT needing any ds_bpermute / permlane reshuffle. The
// fp32 workspace load also improves: VEC=8 loads 2 dwordx4 at a 32B lane
// stride (vs VEC=16's 4 dwordx4 at 64B), so each 256B read transaction is
// half- rather than quarter-utilized.
//
// ── Compile-time split-K (triton MAX_KSPLIT idiom) ─────────────────────────
// SPLIT_K_ template param: when > 0 the split-K accumulation loop bound is a
// compile-time constant and is fully #pragma-unrolled (all workspace loads are
// issued/scheduled up front, matching triton's `tl.arange(0, MAX_KSPLIT)` +
// unrolled sum). SPLIT_K_ == 0 keeps the runtime-`split_k` loop for callers
// that do not specialize. Launchers dispatch the runtime split_k to the
// matching compile-time instantiation (1/2/4/8/...), falling back to 0.
//
// Grid: (ceil(N, VEC * BLOCK), batch * M, 1).
#pragma once

#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a16w16_gfx1250.cuh"  // opus_splitk_ws_handle
#include <cstdint>

template <int VEC_         = 8,
          int BLOCK_       = 64,
          typename D_OUT   = __bf16,
          bool HAS_BIAS_   = false,
          typename D_BIAS_ = D_OUT,
          bool HAS_OOB_    = true,
          int SPLIT_K_     = 0,
          typename D_WS_   = float>
__global__ void splitk_reduce_kernel_gfx1250(const opus_splitk_ws_handle* __restrict__ ws_handle,
                                             D_OUT* __restrict__ c_out,
                                             int split_k,
                                             int M,
                                             int N,
                                             int batch,
                                             int padded_M,
                                             int padded_N,
                                             const D_BIAS_* __restrict__ bias = nullptr,
                                             int bias_stride_batch            = 0)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx1250__)
    // Workspace dtype (partial sums). fp32 = full precision; bf16 = half the
    // split-K-dominated READ traffic (reduce reads split_k slices), summed in
    // fp32 here. The main GEMM must write the workspace in the SAME dtype.
    using D_WS                         = D_WS_;
    const D_WS* __restrict__ workspace = reinterpret_cast<const D_WS*>(ws_handle->ptr);
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
    static_assert(VEC == 8 || VEC == 16,
                  "reduce tail decomposition is specialized for VEC=8 or VEC=16");
    static_assert(!HAS_BIAS || sizeof(D_BIAS) == 2 || sizeof(D_BIAS) == 4,
                  "splitk_reduce HAS_BIAS path supports only 2B or 4B D_BIAS (bf16 / fp32)");

    const int bm_id  = int(opus::block_id_y());
    const int nblk   = int(opus::block_id_x());
    const int tid    = int(opus::thread_id_x());
    const int n_base = (nblk * BLOCK + tid) * VEC;

    const int b = bm_id / M;
    const int m = bm_id - b * M;

    // ── Alternative layout: LOAD-contiguous (no permute, both sides coalesced) ──
    // Instead of "lane owns VEC contiguous N" (store-contig: 1 dwordx4 store fully
    // coalesced, but the fp32 loads are 16B @ VEC*4B stride), use a cyclic-by-4
    // layout: each lane reads a CONTIGUOUS dwordx4 of fp32 per group so the 32
    // lanes of a wave read 512B contiguous (100% read-transaction efficiency).
    // The matching store is 4 D_OUT elems per group -> dwordx2 (bf16, 8B) or
    // dwordx4 (fp32, 16B), and the 32 lanes write 256B / 512B contiguous. This
    // trades one 512B dwordx4 bf16 store for two 256B dwordx2 stores, but makes
    // the split-K-dominated LOAD side fully coalesced. No cross-lane shuffle.
#ifndef OPUS_REDUCE_LOAD_CONTIG
#define OPUS_REDUCE_LOAD_CONTIG 0
#endif
    if constexpr(OPUS_REDUCE_LOAD_CONTIG)
    {
        constexpr int WAVE      = 32;
        constexpr int GROUPS    = VEC / 4; // dwordx4 groups per lane
        const int lane          = tid & (WAVE - 1);
        const int wave_in_blk   = tid >> 5;
        const int wave_n0       = (nblk * (BLOCK / WAVE) + wave_in_blk) * (WAVE * VEC);
        const long split_stride = (long)batch * padded_M * padded_N;
        auto g_ws =
            opus::make_gmem(workspace, (unsigned int)(split_stride * split_k * sizeof(D_WS)));
        auto g_c = opus::make_gmem(c_out, (unsigned int)((size_t)batch * M * N * sizeof(D_OUT)));

        const D_BIAS* bias_base_ptr = HAS_BIAS ? (bias + b * bias_stride_batch) : nullptr;
        auto g_bias                 = opus::make_gmem(
            bias_base_ptr,
            (unsigned int)((bias_stride_batch ? bias_stride_batch : N) * sizeof(D_BIAS)));

#pragma unroll
        for(int g = 0; g < GROUPS; ++g)
        {
            const int n   = wave_n0 + g * (WAVE * 4) + lane * 4; // start N of this 4-wide group
            const int ws0 = b * padded_M * padded_N + m * padded_N + n;
            opus::vector_t<float, 4> a4;
#pragma unroll
            for(int j = 0; j < 4; ++j)
                a4[j] = 0.0f;
            auto acc_slice = [&](int s) {
                auto v4 = g_ws.template load<4>(ws0 + (int)(s * split_stride));
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    a4[j] += static_cast<float>(v4[j]);
            };
            if constexpr(SPLIT_K_ > 0)
            {
#pragma unroll
                for(int s = 0; s < SPLIT_K_; ++s)
                    acc_slice(s);
            }
            else
            {
                for(int s = 0; s < split_k; ++s)
                    acc_slice(s);
            }
            if constexpr(HAS_BIAS)
            {
                auto bv4 = g_bias.template load<4>(n);
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    a4[j] += static_cast<float>(bv4[j]);
            }
            opus::vector_t<D_OUT, 4> o4;
#pragma unroll
            for(int j = 0; j < 4; ++j)
                o4[j] = static_cast<D_OUT>(a4[j]);
            const int c0 = b * M * N + m * N + n;
            if(!HAS_OOB || n + 4 <= N)
            {
                g_c.template store<4>(o4,
                                      c0); // dwordx2 (bf16) / dwordx4 (fp32), 256B/512B contiguous
            }
            else if(n < N)
            {
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    if(n + j < N)
                        g_c.template store<1>(o4[j], c0 + j);
            }
        }
        return;
    }

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

    auto g_ws = opus::make_gmem(workspace, (unsigned int)(split_stride * split_k * sizeof(D_WS)));

    opus::vector_t<float, VEC> acc;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) acc[t] = 0.0f;

    // Accumulate one split-K slice into acc.
    auto accumulate = [&](int s) {
        int ws_idx = ws_row_base + (int)(s * split_stride);
#pragma unroll
        for(int g = 0; g < VEC / 4; ++g)
        {
            auto v4 = g_ws.template load<4>(ws_idx + g * 4);
#pragma unroll
            for(int j = 0; j < 4; ++j)
                acc[g * 4 + j] += static_cast<float>(v4[j]);
        }
    };

    if constexpr(SPLIT_K_ > 0)
    {
// Compile-time split-K: fully-unrolled, no loop-bound register (triton
// MAX_KSPLIT). split_k arg is assumed == SPLIT_K_ by the launcher.
#pragma unroll
        for(int s = 0; s < SPLIT_K_; ++s)
            accumulate(s);
    }
    else
    {
        for(int s = 0; s < split_k; ++s)
            accumulate(s);
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

    // Fast path: whole VEC chunk is in-row -> VEC/STEP contiguous dwordx4 stores.
    // For VEC=8 (STEP=8 bf16) this is exactly ONE buffer_store_dwordx4 per lane,
    // fully coalesced across the wave (see file header).
    auto store_full = [&]() {
        opus::static_for<VEC / STEP>([&](auto g_c_idx) {
            constexpr int g = decltype(g_c_idx)::value;
            g_c.template store<STEP>(slice(out, number<g * STEP>{}, number<(g + 1) * STEP>{}),
                                     c_idx + g * STEP);
        });
    };

    if constexpr (!HAS_OOB) {
        if(n_base + VEC <= N)
            store_full();
    } else {
        if (n_base + VEC <= N) {
            store_full();
        } else if (n_base < N) {
            // Tail path: decompose valid in [1, VEC-1] into descending
            // power-of-2 chunks -> dwordx4 / dwordx2 / dword / short.
            const int valid = N - n_base;
            if constexpr(VEC == 16)
            {
                if constexpr(sizeof(D_OUT) == 2)
                {
                    switch(valid)
                    {
                    case 1: OPUS_REDUCE_ST1(0); break;
                    case 2: OPUS_REDUCE_ST2(0); break;
                    case 3:
                        OPUS_REDUCE_ST2(0);
                        OPUS_REDUCE_ST1(2);
                        break;
                    case 4: OPUS_REDUCE_ST4(0); break;
                    case 5:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST1(4);
                        break;
                    case 6:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST2(4);
                        break;
                    case 7:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST2(4);
                        OPUS_REDUCE_ST1(6);
                        break;
                    case 8: OPUS_REDUCE_ST8(0); break;
                    case 9:
                        OPUS_REDUCE_ST8(0);
                        OPUS_REDUCE_ST1(8);
                        break;
                    case 10:
                        OPUS_REDUCE_ST8(0);
                        OPUS_REDUCE_ST2(8);
                        break;
                    case 11:
                        OPUS_REDUCE_ST8(0);
                        OPUS_REDUCE_ST2(8);
                        OPUS_REDUCE_ST1(10);
                        break;
                    case 12:
                        OPUS_REDUCE_ST8(0);
                        OPUS_REDUCE_ST4(8);
                        break;
                    case 13:
                        OPUS_REDUCE_ST8(0);
                        OPUS_REDUCE_ST4(8);
                        OPUS_REDUCE_ST1(12);
                        break;
                    case 14:
                        OPUS_REDUCE_ST8(0);
                        OPUS_REDUCE_ST4(8);
                        OPUS_REDUCE_ST2(12);
                        break;
                    case 15:
                        OPUS_REDUCE_ST8(0);
                        OPUS_REDUCE_ST4(8);
                        OPUS_REDUCE_ST2(12);
                        OPUS_REDUCE_ST1(14);
                        break;
                    }
                }
                else
                {
                    switch(valid)
                    {
                    case 1: OPUS_REDUCE_ST1(0); break;
                    case 2: OPUS_REDUCE_ST2(0); break;
                    case 3:
                        OPUS_REDUCE_ST2(0);
                        OPUS_REDUCE_ST1(2);
                        break;
                    case 4: OPUS_REDUCE_ST4(0); break;
                    case 5:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST1(4);
                        break;
                    case 6:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST2(4);
                        break;
                    case 7:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST2(4);
                        OPUS_REDUCE_ST1(6);
                        break;
                    case 8:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST4(4);
                        break;
                    case 9:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST4(4);
                        OPUS_REDUCE_ST1(8);
                        break;
                    case 10:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST4(4);
                        OPUS_REDUCE_ST2(8);
                        break;
                    case 11:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST4(4);
                        OPUS_REDUCE_ST2(8);
                        OPUS_REDUCE_ST1(10);
                        break;
                    case 12:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST4(4);
                        OPUS_REDUCE_ST4(8);
                        break;
                    case 13:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST4(4);
                        OPUS_REDUCE_ST4(8);
                        OPUS_REDUCE_ST1(12);
                        break;
                    case 14:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST4(4);
                        OPUS_REDUCE_ST4(8);
                        OPUS_REDUCE_ST2(12);
                        break;
                    case 15:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST4(4);
                        OPUS_REDUCE_ST4(8);
                        OPUS_REDUCE_ST2(12);
                        OPUS_REDUCE_ST1(14);
                        break;
                    }
                }
            }
            else
            { // VEC == 8: STEP=8 (bf16) or STEP=4 (fp32); valid in [1,7]
                switch (valid) {
                    case  1: OPUS_REDUCE_ST1( 0); break;
                    case  2: OPUS_REDUCE_ST2( 0); break;
                    case  3: OPUS_REDUCE_ST2( 0); OPUS_REDUCE_ST1( 2); break;
                    case  4: OPUS_REDUCE_ST4( 0); break;
                    case  5: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST1( 4); break;
                    case  6: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST2( 4); break;
                    case 7:
                        OPUS_REDUCE_ST4(0);
                        OPUS_REDUCE_ST2(4);
                        OPUS_REDUCE_ST1(6);
                        break;
                    }
            }
        }
        // else: whole VEC chunk is past N -> write nothing.
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
