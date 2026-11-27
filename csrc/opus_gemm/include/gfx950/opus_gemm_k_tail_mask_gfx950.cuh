// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Register-level K-tail mask shared by the gfx950 a16w16 pipelines
// (ROCm/aiter#5954).
//
// The A/B buffer resources are bounded by rows, not by K. When
// K % B_K != 0 the last K tile therefore reads the bytes that follow logical
// K in each physical row (padding, or the next row when stride == K). Those
// bytes are garbage as far as the GEMM is concerned, and a NaN/Inf among them
// poisons the accumulator even when the other operand is zero (0 * NaN).
//
// The fix is applied after the last K tile has been read from LDS into the
// MFMA operand registers and before its MMA: every element of the A *and* B
// fragments with k >= K is overwritten with zero. Callers guard the call with a
// wave-uniform `if (kargs.k % B_K != 0)` so aligned K pays only a scalar
// compare + branch, and only in the epilogue (never in the main loop).
//
// K must be even (all a16w16 launchers reject odd K), so the mask works on
// 32-bit pairs of 16-bit elements.
//
// Fragment layout (identical for the A and B operands of every pipeline that
// uses this header; see make_layout_r{a,b}* in the pipeline headers):
//   v[idx], idx = (i_x * E_K + i_k) * PACK + j
//     i_x in [0, E_X) repetition along M (A) or N (B)
//     i_k in [0, E_K) MFMA K-steps inside one B_K tile
//     j   in [0, PACK) contiguous K elements held by one lane (VEC_A/VEC_B)
//   K inside the tile = i_k * W_K + (lane_id / W_X) * PACK + j
// The 16x16x32 (W_X=16, W_K=32, PACK=8) and 32x32x16 (W_X=32, W_K=16,
// PACK=8) bf16/fp16 MFMAs both satisfy W_X * W_K == WARP_SIZE * PACK, which is
// what makes the single-pass K mapping above exact.
#pragma once

#include <opus/opus.hpp>

#ifdef __HIP_DEVICE_COMPILE__

namespace opus_k_tail_gfx950 {

template<int E_X, int E_K, int W_X, int W_K, int PACK, typename V>
__device__ __attribute__((always_inline)) inline
void mask_k_tail_frag(V& v, int k_valid, int lane_id) {
    using elem_t = opus::remove_cvref_t<decltype(v[0])>;
    static_assert(W_X * W_K == 64 * PACK,
                  "K-tail mask assumes one MFMA K step per lane group (R == 1)");
    static_assert(sizeof(elem_t) == 2 && PACK % 2 == 0,
                  "K-tail mask works on 32-bit pairs of 16-bit elements");
    static_assert(opus::size<V>() == E_X * E_K * PACK,
                  "K-tail mask: fragment size does not match E_X * E_K * PACK");
    // Work on 32-bit pairs: K is even (every a16w16 launcher rejects odd K) and
    // B_K, W_K, PACK are even, so a pair (k, k + 1) is either entirely inside
    // or entirely outside logical K. One v_cndmask_b32 per pair, with the
    // lane-dependent compare shared by every E_X repetition, keeps the extra
    // register pressure to a few SGPR masks (the persistent kernels have
    // almost no VGPR headroom).
    constexpr int PAIRS = PACK / 2;
    using pair_vec_t = opus::u32_t __attribute__((ext_vector_type(E_X * E_K * PAIRS)));
    auto w = __builtin_bit_cast(pair_vec_t, v);
    const int k_group = lane_id / W_X;
    opus::static_for<E_K>([&](auto ik_c) {
        constexpr int ik = decltype(ik_c)::value;
        opus::static_for<PAIRS>([&](auto d_c) {
            constexpr int d = decltype(d_c)::value;
            // first K of this pair inside the B_K tile
            const bool inside = (ik * W_K + 2 * d + k_group * PACK) < k_valid;
            opus::static_for<E_X>([&](auto ix_c) {
                constexpr int ix = decltype(ix_c)::value;
                constexpr int idx = (ix * E_K + ik) * PAIRS + d;
                w[idx] = inside ? w[idx] : opus::u32_t{0};
            });
        });
    });
    v = __builtin_bit_cast(V, w);
}

// Convenience wrappers for traits that expose E_M/E_N/E_K/W_M/W_N/W_K and
// VEC_A/VEC_B (split-barrier, persistent, mono-tile and their 4g_safe forms).
template<typename T, typename V>
__device__ __attribute__((always_inline)) inline
void mask_k_tail_a(V& v_a, int k_valid, int lane_id) {
    mask_k_tail_frag<T::E_M, T::E_K, T::W_M, T::W_K, T::VEC_A>(v_a, k_valid, lane_id);
}

template<typename T, typename V>
__device__ __attribute__((always_inline)) inline
void mask_k_tail_b(V& v_b, int k_valid, int lane_id) {
    mask_k_tail_frag<T::E_N, T::E_K, T::W_N, T::W_K, T::VEC_B>(v_b, k_valid, lane_id);
}

} // namespace opus_k_tail_gfx950

#endif // __HIP_DEVICE_COMPILE__
