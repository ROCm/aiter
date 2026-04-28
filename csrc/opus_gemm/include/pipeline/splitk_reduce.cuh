// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Split-K reduce kernel: tile-agnostic; sums an fp32 workspace across the
// split-K axis, casts fp32 -> D_OUT, and writes to C. Split out of
// opus_gemm_pipeline_a16w16_flatmm_splitk.cuh so the reduce path can be
// shared by future split-K main pipelines (a8w8 etc.) without dragging in
// the full a16w16 flatmm pipeline header.
//
// Template parameters:
//   * VEC_   - elements per thread along N (fast-path tile width).
//   * BLOCK_ - threads per block.
//   * D_OUT  - output element type. Currently exercised with __bf16 and float;
//              any other 16-bit / 32-bit type that opus::vector_t supports
//              also works mechanically because the store path dispatches on
//              sizeof(D_OUT).
//
// All splitk launchers invoke this kernel with <VEC_=16, BLOCK_=64>; the
// D_OUT defaults to __bf16 to keep call-sites that omit the type unchanged.
//
// Grid: (ceil(N, VEC * BLOCK), batch * M, 1).
// Each thread handles VEC fp32 lanes along N; the workspace load path is
// always fp32 (4x buffer_load_dwordx4 for VEC=16) regardless of D_OUT.
//
// Store path bytes-per-thread:
//   * D_OUT = __bf16 -> 16 elems x 2B = 32B   (2 x buffer_store_dwordx4)
//   * D_OUT = float  -> 16 elems x 4B = 64B   (4 x buffer_store_dwordx4)
// We pick STEP = 16 / sizeof(D_OUT) so each store covers exactly one
// dwordx4 (128-bit) chunk and the inner loop runs VEC / STEP times.
//
// Store-path 3-way split on the N tail (unchanged from the bf16-only
// implementation):
//   * (n_base + VEC <= N): fast path, VEC/STEP x buffer_store_dwordx4.
//   * (n_base < N):        tail path, VEC_valid scalar buffer_store_b16/b32
//                          per in-range element. Prevents a 128-bit vector
//                          store from straddling the row-N boundary and
//                          silently landing in the next row of the
//                          row-major C tensor (the buffer rsrc only checks
//                          linear byte offset, not per-row column bounds).
//   * (n_base >= N):       skip entirely.
#pragma once

#include "../opus_gemm_utils.cuh"

template<int VEC_ = 16, int BLOCK_ = 64, typename D_OUT = __bf16>
__global__ void splitk_reduce_kernel(
    const float* __restrict__ workspace,
    D_OUT*       __restrict__ c_out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N)
{
#ifdef __HIP_DEVICE_COMPILE__
    constexpr int VEC   = VEC_;
    constexpr int BLOCK = BLOCK_;

    // STEP = elements per buffer_store_dwordx4. STEP * sizeof(D_OUT) == 16.
    constexpr int STEP = 16 / sizeof(D_OUT);
    static_assert(STEP * sizeof(D_OUT) == 16,
                  "D_OUT must divide a 128-bit store boundary cleanly "
                  "(supported sizes: 2B / 4B; e.g. __bf16, float)");
    static_assert(VEC % STEP == 0,
                  "VEC must be a multiple of STEP so the fast path tiles "
                  "into whole dwordx4 stores");

    const int bm_id  = int(opus::block_id_y());            // 0..batch*M-1
    const int nblk   = int(opus::block_id_x());
    const int tid    = int(opus::thread_id_x());
    const int n_base = (nblk * BLOCK + tid) * VEC;

    const int b = bm_id / M;
    const int m = bm_id - b * M;

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

    opus::vector_t<D_OUT, VEC> out;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) out[t] = static_cast<D_OUT>(acc[t]);

    auto g_c = opus::make_gmem(c_out,
                               (unsigned int)((size_t)batch * M * N * sizeof(D_OUT)));
    const int c_idx = b * M * N + m * N + n_base;  // in D_OUT elements

    if (n_base + VEC <= N) {
        // Fast path: entire VEC chunk is in-row -> VEC/STEP x buffer_store_dwordx4.
        // static_for promotes the loop index to constexpr so opus::slice's
        // compile-time bounds (opus::number<>) can be formed.
        opus::static_for<VEC / STEP>([&](auto g_c_idx) {
            constexpr int g = decltype(g_c_idx)::value;
            g_c.template store<STEP>(
                opus::slice(out,
                            opus::number<g * STEP>{},
                            opus::number<(g + 1) * STEP>{}),
                c_idx + g * STEP);
        });
    } else if (n_base < N) {
        // Tail path: only the first `valid` elements of the VEC chunk are in
        // row m. Scalar-store them one by one; the remaining (VEC - valid)
        // elements would otherwise spill into row m+1 via a 128-bit vector
        // store (buffer rsrc cannot catch this because the target byte offset
        // is still < rsrc.size in a row-major tensor).
        const int valid = N - n_base;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            if (j < valid) g_c.template store<1>(out[j], c_idx + j);
        }
    }
    // else: whole VEC chunk is past N -> write nothing.
#endif  // __HIP_DEVICE_COMPILE__
}
