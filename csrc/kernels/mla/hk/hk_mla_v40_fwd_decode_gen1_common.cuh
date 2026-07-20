// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared core for the V4.0 gen.1 MLA decode kernels (m16x8 / m16x4): the pinned
// VGPR register map (HkMlaV40Regs) and the PV GEMM stage (hk_mla_v40_pv_gemm /
// hk_mla_v40_pv_stage). Identical per-wave for both head counts, so they live
// here once. WarpType stays per-kernel (the m16x4 variant has more states).
#pragma once

#include "hk_mla_softmax.cuh"
#include "hk_mla_v40_buffer_managers_gen1.cuh"

using namespace hk_mla;

// ---- Shared pinned-VGPR register map (per-lane) ----
// Single source of truth for the hand-pinned VGPR layout + art (auto-register
// tile) range/type views, so the per-tile stage functions and both per-tile
// orchestrators (even/odd) bind to the *same* physical registers. art tiles are
// stateless register views (the binding is the type's range param), so a stage
// reconstructs `typename R::p_comp_t p_comp;` and operates on the same vgprs the
// orchestrator's clobber reserved. See the original inline map for the rationale
// of every offset (3-register QK K + overlay reuse, v64..v67 unpinned gap).
template <typename T>
struct HkMlaV40Regs
{
    using comp_t    = float;
    using mfma_ab_t = hk::bf16;

    // p_comp = kBlockN N-cols x kTileM(16) rows / 64 lanes = kBlockN/4 fp32/lane;
    // p_mfma is its bf16-packed half. kBlockN=32 -> 8/4 (m16x4, m16x8 legacy);
    // kBlockN=64 -> 16/8 (m16x8 double-tile: sub-tile A in the low half, B high).
    static constexpr uint32_t k_o_sz       = 128;
    static constexpr uint32_t k_p_comp_sz  = T::kBlockN / 4u;
    static constexpr uint32_t k_p_mfma_sz  = T::kBlockN / 8u;
    static constexpr uint32_t k_q_vgpr_sz  = 64; // full Q block (512 cols)
    static constexpr uint32_t mfma_tile_sz = 4;  // one 16x32 bf16 base tile
    // Per-iteration count of 32-row KV sub-tiles (1 for kBlockN=32, 2 for 64).
    static constexpr uint32_t kNumKvSub = T::kBlockN / 32u;

    // # of QK col-tiles whose Q comes from the contiguous q_vgpr block (the rest
    // come from LDS). Migration knob: 10 (0:320), 12 (0:384), 14 (all NoPE),
    // 16 (all NoPE + RoPE -> NOTHING from LDS in the QK loop).
    static constexpr uint32_t kQkGemmTiles = 16;
    // RoPE (col-tiles 14,15 / cols 448:512) lives in VGPR (read from Q-LDS in the
    // prologue) iff the QK loop sources all 16 col-tiles from VGPR.
    static constexpr bool kRopeInVgpr = (kQkGemmTiles >= 16u);

    static constexpr uint32_t k_o_end        = 255;
    static constexpr uint32_t k_o_begin      = k_o_end - k_o_sz + 1;             // 128
    static constexpr uint32_t k_q_vgpr_end   = k_o_begin - 1;                    // 127
    static constexpr uint32_t k_q_vgpr_begin = k_q_vgpr_end - k_q_vgpr_sz + 1;   // 64
    static constexpr uint32_t k_p_comp_end   = k_q_vgpr_begin - 1;               // 63
    static constexpr uint32_t k_p_comp_begin = k_p_comp_end - k_p_comp_sz + 1;   // 56 / 48
    static constexpr uint32_t k_p_mfma_begin = k_p_comp_begin + 0;               // overlay
    static constexpr uint32_t k_p_mfma_end   = k_p_mfma_begin + k_p_mfma_sz - 1; // 59 / 55
    // pv_v_0 (QK-unused, PV-only) sits at the top 4 of the p_comp region (dead
    // during PV). 60:63 for both kBlockN (== p_comp_begin+4 at 32).
    static constexpr uint32_t k_v0_begin = k_p_comp_end - 3;              // 60
    static constexpr uint32_t k_v0_end   = k_v0_begin + mfma_tile_sz - 1; // 63
    // 3-slot QK K ring, 12 regs directly below p_comp: [p_comp_begin-12, -1].
    //   kBlockN=32: k0=52,k1=48,k2=44 (original, top-down).
    //   kBlockN=64: k0=36,k1=40,k2=44 (bottom-up) so p_comp can grow to 48:63.
    // Only k0<->k2 swap vs the linear order; k1 is the midpoint either way, so the
    // 32 case is byte-identical to the pre-double map.
    static constexpr uint32_t k_k0_begin =
        (T::kBlockN == 32) ? (k_p_comp_begin - mfma_tile_sz) : (k_p_comp_begin - 3u * mfma_tile_sz);
    static constexpr uint32_t k_k1_begin = k_p_comp_begin - 2u * mfma_tile_sz;
    static constexpr uint32_t k_k2_begin =
        (T::kBlockN == 32) ? (k_p_comp_begin - 3u * mfma_tile_sz) : (k_p_comp_begin - mfma_tile_sz);
    // Lowest pinned VGPR -> compiler scratch budget = [0, k_scratch_budget).
    static constexpr uint32_t k_scratch_budget = k_p_comp_begin - 3u * mfma_tile_sz; // 44 / 36
    // q_lds (Phase-B Q-from-LDS scratch) reuses the unused top of q_vgpr, right
    // after the kQkGemmTiles VGPR col-tiles: v[64 + 4*kQkGemmTiles .. +7].
    static constexpr uint32_t k_q_lds_begin   = k_q_vgpr_begin + 4u * kQkGemmTiles;
    static constexpr uint32_t k_q_lds_0_begin = k_q_lds_begin;
    static constexpr uint32_t k_q_lds_1_begin = k_q_lds_begin + mfma_tile_sz;

    using q_vgpr_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_q_vgpr_begin, k_q_vgpr_end>>, 4>;
    using p_comp_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_comp_begin, k_p_comp_end>>, 4>;
    using p_comp_lo_ranges = hkdart::
        split_many_t<hkdart::type_list<hkdart::range<k_p_comp_begin + 0, k_p_comp_begin + 3>>, 4>;
    using p_comp_hi_ranges = hkdart::
        split_many_t<hkdart::type_list<hkdart::range<k_p_comp_begin + 4, k_p_comp_begin + 7>>, 4>;
    // Sub-tile B N-groups (kv-rows 32:48, 48:64), only for kBlockN=64. For 32 these
    // alias regs above p_comp (unused; m16x4 never references the b_* aliases).
    using p_comp_b_lo_ranges = hkdart::
        split_many_t<hkdart::type_list<hkdart::range<k_p_comp_begin + 8, k_p_comp_begin + 11>>, 4>;
    using p_comp_b_hi_ranges = hkdart::
        split_many_t<hkdart::type_list<hkdart::range<k_p_comp_begin + 12, k_p_comp_begin + 15>>, 4>;
    using kv_top_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_k0_begin, k_k0_begin + 3>>, 4>;
    using kv_bot_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_k1_begin, k_k1_begin + 3>>, 4>;
    using kv_alt_top_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_k2_begin, k_k2_begin + 3>>, 4>;
    using p_mfma_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_mfma_begin, k_p_mfma_end>>, 4>;
    // Per-sub-tile p_mfma halves (16x32 = 4 dwords each): A = packed p_comp N 0:32,
    // B = N 32:64. For kBlockN=32 only A is used (== p_mfma); B aliases unused regs.
    using p_mfma_a_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_mfma_begin, k_p_mfma_begin + 3>>,
                             4>;
    using p_mfma_b_ranges = hkdart::
        split_many_t<hkdart::type_list<hkdart::range<k_p_mfma_begin + 4, k_p_mfma_begin + 7>>, 4>;
    using o_ranges = hkdart::split_many_t<hkdart::type_list<hkdart::range<k_o_begin, k_o_end>>, 4>;
    using pv_v_0_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_v0_begin, k_v0_end>>, 4>;
    using pv_v_1_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_k0_begin, k_k0_begin + 3>>, 4>;
    using pv_v_2_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_k1_begin, k_k1_begin + 3>>, 4>;
    // 4th PV V slot reuses the k_2 regs (v44:47) -- free during PV -- so the
    // round-robin refill ds_read can target a different reg than the MFMA it is
    // issued behind (breaks the ds_read-dst == mfma-src WAR stall).
    using pv_v_3_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_k2_begin, k_k2_begin + 3>>, 4>;
    using q_lds_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_q_lds_begin, k_q_lds_begin + 7>>, 4>;
    // art tile types (stage functions reconstruct these from the shared ranges).
    using q_vgpr_t = hk::art<mfma_ab_t, T::kTileM, 512, hk::row_l, hk::rt_16x32_s, q_vgpr_ranges>;
    using p_comp_t =
        hk::art<comp_t, T::kBlockN, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_ranges>;
    using p_comp_lo_t = hk::art<comp_t, 16, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_lo_ranges>;
    using p_comp_hi_t = hk::art<comp_t, 16, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_hi_ranges>;
    using p_comp_b_lo_t =
        hk::art<comp_t, 16, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_b_lo_ranges>;
    using p_comp_b_hi_t =
        hk::art<comp_t, 16, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_b_hi_ranges>;
    using k_0_t =
        hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, kv_top_ranges>;
    using k_1_t =
        hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, kv_bot_ranges>;
    using k_2_t =
        hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, kv_alt_top_ranges>;
    using pv_v_0_t =
        hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, pv_v_0_ranges>;
    using pv_v_1_t =
        hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, pv_v_1_ranges>;
    using pv_v_2_t =
        hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, pv_v_2_ranges>;
    using pv_v_3_t =
        hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, pv_v_3_ranges>;
    using p_mfma_t =
        hk::art<mfma_ab_t, T::kTileM, T::kBlockN, hk::row_l, hk::rt_16x32_s, p_mfma_ranges>;
    // 16x32 per-sub-tile P (bf16) for the PV GEMM; PV runs once per sub-tile.
    using p_mfma_a_t =
        hk::art<mfma_ab_t, T::kTileM, 32, hk::row_l, hk::rt_16x32_s, p_mfma_a_ranges>;
    using p_mfma_b_t =
        hk::art<mfma_ab_t, T::kTileM, 32, hk::row_l, hk::rt_16x32_s, p_mfma_b_ranges>;
    using oaccu_t = hk::art<comp_t, T::kTileM, T::kVoHeadDim, hk::row_l, hk::rt_16x16_s, o_ranges>;
};

// ---- PV GEMM stage ----
//
// O = P @ V computed as oaccu^T = V^T @ P^T via mma_ABt(oaccu, V, p_mfma).
// V streams as 32 base tiles S_0..S_31 through a 3-deep round-robin
// pv_v_0/pv_v_1/pv_v_2 (S_j -> slot j%3, 6 ds_read_b64 in flight). p_lds_v is
// the V pong (curr-pong; PV runs at call end). kDoRescale folds the
// online-softmax oaccu rescale; kIsFirstIter inits oaccu fresh (3-arg mma) and
// never rescales.
// PMfmaT is the 16x32 per-sub-tile P (4 dwords): p_mfma_a_t (default; also == the
// full p_mfma for kBlockN=32) or p_mfma_b_t (kBlockN=64 sub-tile B). One PV call
// contracts one 32-wide KV sub-tile; kBlockN=64 calls this twice (A then B).
// kRowBase: V sub-tile row base (0 = sub-tile A, 32 = sub-tile B). Folded into the
// V ds_read imm by load_transposed_v_to_gpr so the caller passes the pong BASE.
template <bool kIsFirstIter, bool kDoRescale, typename T>
__device__ __forceinline__ void hk_mla_v40_pv_gemm(KvManager8to16bitsV1<T>& kv_manager,
                                                   const uintptr_t p_lds_v,
                                                   const float rescale)
{
    using R                       = HkMlaV40Regs<T>;
    using comp_t                  = typename R::comp_t;
    constexpr uint32_t k_o_begin  = R::k_o_begin;
    constexpr uint32_t k_v0_begin = R::k_v0_begin;
    constexpr uint32_t k_k0_begin = R::k_k0_begin;
    constexpr uint32_t k_k1_begin = R::k_k1_begin;
    constexpr uint32_t k_k2_begin = R::k_k2_begin;
    // kBlockN=64 (m16x8 double-tile) contracts BOTH 32-row sub-tiles in one call:
    // sub-tile A (rows 0:32) uses p_mfma_a, B (rows 32:64) uses p_mfma_b. kBlockN=32
    // has only A. Picked per-iter by row_base (see below), no caller arg.
    typename R::p_mfma_a_t p_mfma_a;
    typename R::p_mfma_b_t p_mfma_b;
    typename R::pv_v_0_t pv_v_0;
    typename R::pv_v_1_t pv_v_1;
    typename R::pv_v_2_t pv_v_2;
    typename R::pv_v_3_t pv_v_3;

    // D-tiling: each D-iter emits 2 oaccu 16x16 sub-tiles (32 D-cols); 512/32 = 16
    // D-iters per 32-row KV sub-tile. kBlockN=64 runs BOTH sub-tiles in one call, so
    // num_pv_iter = kNumKvSub * 16. iter [0,16) = sub-tile A, [16,32) = sub-tile B.
    constexpr uint32_t kDIters     = T::kVoHeadDim / (2u * T::kTileM); // 16
    constexpr uint32_t num_pv_iter = R::kNumKvSub * kDIters;           // 16 or 32
    // Flat V base tiles across all sub-tiles: 32 per sub-tile (S_0..S_31 each).
    constexpr uint32_t kVTilesPerSub = 2u * kDIters;                 // 32
    constexpr uint32_t kNumVTiles    = R::kNumKvSub * kVTilesPerSub; // 32 or 64

    auto pk_mul_pair = [&](float r, auto base_c) {
        constexpr uint32_t base = decltype(base_c)::value;
        const float2 r2         = {r, r};
        asm volatile("v_pk_mul_f32 v[%0:%1], %2, v[%0:%1]" : : "n"(base), "n"(base + 1), "v"(r2));
    };
    auto mul_pair = [&](float r, auto base_c) {
        constexpr uint32_t base = decltype(base_c)::value;
        asm volatile("v_mul_f32_e32 v[%0], %1, v[%0]" : : "n"(base), "v"(r));
        asm volatile("v_mul_f32_e32 v[%0], %1, v[%0]" : : "n"(base + 1), "v"(r));
    };

    // Issue both ds_read_b64 for base tile S_jj (within its sub-tile) into a
    // round-robin slot (4 slots). kRowBase selects the KV sub-tile (0=A, 32=B),
    // folded into the V ds_read imm by load_transposed_v_to_gpr.
    auto load_S = [&]<uint32_t kRowBase, uint32_t jj, uint32_t slot>() {
        constexpr uint32_t base = (slot == 0u)   ? k_v0_begin
                                  : (slot == 1u) ? k_k0_begin
                                  : (slot == 2u) ? k_k1_begin
                                                 : k_k2_begin;
        kv_manager.template load_transposed_v_to_gpr<kRowBase + 0u, jj * 16u, base + 0>(p_lds_v);
        kv_manager.template load_transposed_v_to_gpr<kRowBase + 16u, jj * 16u, base + 2>(p_lds_v);
    };
    // mfma: oaccu_dst (+)= pv_v_{slot}^T @ p_mfma[sub-tile]. 3-arg init only on the
    // FIRST sub-tile of a first-iter call (kRowBase==0); sub-tile B always accumulates
    // onto A's oaccu (same oaccu tile, second N-half of P).
    auto do_mma = [&]<uint32_t kRowBase, uint32_t slot, typename OA>(OA& oaccu_dst) {
        auto run = [&](auto& v) {
            if constexpr(kIsFirstIter && kRowBase == 0u)
                hk::mma_ABt(oaccu_dst, v, p_mfma_a);
            else if constexpr(kRowBase == 0u)
                hk::mma_ABt(oaccu_dst, v, p_mfma_a, oaccu_dst);
            else
                hk::mma_ABt(oaccu_dst, v, p_mfma_b, oaccu_dst);
        };
        if constexpr(slot == 0u)
            run(pv_v_0);
        else if constexpr(slot == 1u)
            run(pv_v_1);
        else if constexpr(slot == 2u)
            run(pv_v_2);
        else
            run(pv_v_3);
    };

    if constexpr(kDoRescale)
    {
        opus::static_for<2>([&](auto s) {
            pk_mul_pair(rescale, opus::number<k_o_begin + s.value * 4u + 0u>{});
            pk_mul_pair(rescale, opus::number<k_o_begin + s.value * 4u + 2u>{});
        });
    }

    // Prologue: preload the first 3 flat V tiles (S_0,S_1,S_2 of sub-tile A) into
    // slots 0,1,2. ONE prologue for the whole call -- sub-tile B's S_0..S_2 are
    // prefetched by sub-tile A's tail iters via the flat prefetch index below.
    load_S.template operator()<0u, 0u, 0u>();
    load_S.template operator()<0u, 1u, 1u>();
    load_S.template operator()<0u, 2u, 2u>();

    opus::static_for<num_pv_iter>([&](auto i) {
        constexpr uint32_t iter     = i.value;
        constexpr uint32_t col_idx  = iter % kDIters;         // 0..15 D-tile within sub-tile
        constexpr uint32_t row_base = (iter / kDIters) * 32u; // 0 (A) or 32 (B)
        constexpr bool has_next     = (iter + 1u) < num_pv_iter;
        // Rescale only folds on the FIRST sub-tile pass (row_base==0) and only when a
        // NEXT D-tile in that pass exists (col_idx+1 < kDIters) -- else oaccu overflows.
        constexpr bool resc_next = kDoRescale && (row_base == 0u) && (col_idx + 1u < kDIters);
        constexpr uint32_t next_oaccu_base = k_o_begin + (col_idx + 1u) * 8u;

        // Flat V-tile index across all sub-tiles: prefetch runs on this so sub-tile A's
        // tail prefetches sub-tile B's S_0.. (ONE prologue). Consume uses col_idx.
        constexpr uint32_t jf_lo = 2u * iter;      // flat mfma idx (a)
        constexpr uint32_t jf_hi = 2u * iter + 1u; // flat mfma idx (b)
        // 4-slot round-robin (flat tile T -> slot T%4). Reading slot j%4 while refilling
        // slot (j+3)%4 keeps the refill ds_read's dst != the mfma's src reg.
        constexpr uint32_t slot_lo = jf_lo % 4u;
        constexpr uint32_t slot_hi = jf_hi % 4u;

        constexpr uint32_t oaccu_base = k_o_begin + col_idx * 8u;
        using oaccu_a_r =
            hkdart::split_many_t<hkdart::type_list<hkdart::range<oaccu_base + 0, oaccu_base + 3>>,
                                 4>;
        using oaccu_b_r =
            hkdart::split_many_t<hkdart::type_list<hkdart::range<oaccu_base + 4, oaccu_base + 7>>,
                                 4>;
        hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_a_r> oaccu_a;
        hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_b_r> oaccu_b;

        // Prefetch S at flat index jf+3 -> its own sub-tile row_base + local jj%32.
        auto prefetch = [&]<uint32_t jf>() {
            if constexpr(jf < kNumVTiles)
            {
                constexpr uint32_t pf_row = (jf / kVTilesPerSub) * 32u;
                constexpr uint32_t pf_jj  = jf % kVTilesPerSub;
                load_S.template operator()<pf_row, pf_jj, jf % 4u>();
            }
        };

        // ---- mfma_a (flat jf_lo, reads slot_lo) ----
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(has_next ? 4 : 2, -1));
        prefetch.template operator()<jf_lo + 3u>();
        do_mma.template operator()<row_base, slot_lo>(oaccu_a);
        if constexpr(resc_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 0>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 0>{});
        }

        // ---- mfma_b (flat jf_hi, reads slot_hi) ----
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(has_next ? 4 : 0, -1));
        prefetch.template operator()<jf_hi + 3u>();
        do_mma.template operator()<row_base, slot_hi>(oaccu_b);
        if constexpr(resc_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 2>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 2>{});
        }
    });
}

// PV stage selector: picks the kIsFirstIter / kDoRescale instantiation from the
// runtime do_rescale decision the softmax produced.
template <bool kIsFirstIter, typename T>
__device__ __forceinline__ void hk_mla_v40_pv_stage(KvManager8to16bitsV1<T>& kv_manager,
                                                    const uintptr_t p_lds_v,
                                                    const float rescale,
                                                    const bool do_rescale)
{
    if constexpr(kIsFirstIter)
    {
        hk_mla_v40_pv_gemm<true, false, T>(kv_manager, p_lds_v, rescale);
    }
    else if(do_rescale)
    {
        hk_mla_v40_pv_gemm<false, true, T>(kv_manager, p_lds_v, rescale);
    }
    else
    {
        hk_mla_v40_pv_gemm<false, false, T>(kv_manager, p_lds_v, rescale);
    }
}
