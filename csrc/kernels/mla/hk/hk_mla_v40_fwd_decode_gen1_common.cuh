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

    static constexpr uint32_t k_o_sz      = 128;
    static constexpr uint32_t k_p_comp_sz = 8;
    static constexpr uint32_t k_p_mfma_sz = 4;
    static constexpr uint32_t k_q_vgpr_sz = 64; // full Q block (512 cols)
    static constexpr uint32_t mfma_tile_sz = 4; // one 16x32 bf16 base tile

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
    static constexpr uint32_t k_p_comp_begin = k_p_comp_end - k_p_comp_sz + 1;   // 56
    static constexpr uint32_t k_p_mfma_begin = k_p_comp_begin + 0;              // 56 (overlay)
    static constexpr uint32_t k_p_mfma_end   = k_p_mfma_begin + k_p_mfma_sz - 1; // 59
    static constexpr uint32_t k_v0_begin     = k_p_comp_begin + 4;              // 60
    static constexpr uint32_t k_v0_end       = k_v0_begin + mfma_tile_sz - 1;      // 63
    static constexpr uint32_t k_k0_begin     = k_p_comp_begin - mfma_tile_sz;      // 52
    static constexpr uint32_t k_k1_begin     = k_k0_begin - mfma_tile_sz;          // 48
    static constexpr uint32_t k_k2_begin     = k_k1_begin - mfma_tile_sz;          // 44
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
    using kv_top_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_k0_begin, k_k0_begin + 3>>, 4>;
    using kv_bot_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_k1_begin, k_k1_begin + 3>>, 4>;
    using kv_alt_top_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_k2_begin, k_k2_begin + 3>>, 4>;
    using p_mfma_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_mfma_begin, k_p_mfma_end>>, 4>;
    using o_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_o_begin, k_o_end>>, 4>;
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
    using p_comp_t = hk::art<comp_t, T::kBlockN, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_ranges>;
    using p_comp_lo_t =
        hk::art<comp_t, 16, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_lo_ranges>;
    using p_comp_hi_t =
        hk::art<comp_t, 16, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_hi_ranges>;
    using k_0_t = hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, kv_top_ranges>;
    using k_1_t = hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, kv_bot_ranges>;
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
template <bool kIsFirstIter, bool kDoRescale, typename T>
__device__ __forceinline__ void
hk_mla_v40_pv_gemm(KvManager8to16bitsV1<T>& kv_manager, const uintptr_t p_lds_v, const float rescale)
{
    using R                        = HkMlaV40Regs<T>;
    using comp_t                   = typename R::comp_t;
    constexpr uint32_t k_o_begin   = R::k_o_begin;
    constexpr uint32_t k_v0_begin  = R::k_v0_begin;
    constexpr uint32_t k_k0_begin  = R::k_k0_begin;
    constexpr uint32_t k_k1_begin  = R::k_k1_begin;
    constexpr uint32_t k_k2_begin  = R::k_k2_begin;
    typename R::p_mfma_t p_mfma;
    typename R::pv_v_0_t pv_v_0;
    typename R::pv_v_1_t pv_v_1;
    typename R::pv_v_2_t pv_v_2;
    typename R::pv_v_3_t pv_v_3;

    constexpr uint32_t num_pv_iter = T::kVoHeadDim / T::kBlockN; // 16
    constexpr uint32_t kNumVTiles  = 2u * num_pv_iter;          // 32 = S_0..S_31

    auto pk_mul_pair = [&](float r, auto base_c) {
        constexpr uint32_t base = decltype(base_c)::value;
        const float2 r2         = {r, r};
        asm volatile("v_pk_mul_f32 v[%0:%1], %2, v[%0:%1]"
                     :
                     : "n"(base), "n"(base + 1), "v"(r2));
    };
    auto mul_pair = [&](float r, auto base_c) {
        constexpr uint32_t base = decltype(base_c)::value;
        asm volatile("v_mul_f32_e32 v[%0], %1, v[%0]" : : "n"(base), "v"(r));
        asm volatile("v_mul_f32_e32 v[%0], %1, v[%0]" : : "n"(base + 1), "v"(r));
    };

    // Issue both ds_read_b64 for base tile S_jj into round-robin slot (4 slots).
    auto load_S = [&]<uint32_t jj, uint32_t slot>() {
        constexpr uint32_t base = (slot == 0u)   ? k_v0_begin
                                  : (slot == 1u) ? k_k0_begin
                                  : (slot == 2u) ? k_k1_begin
                                                 : k_k2_begin;
        kv_manager.template load_transposed_v_to_gpr<0u, jj * 16u, base + 0>(p_lds_v);
        kv_manager.template load_transposed_v_to_gpr<16u, jj * 16u, base + 2>(p_lds_v);
    };
    // mfma: oaccu_dst (+)= pv_v_{slot}^T @ p_mfma (3-arg init when first).
    auto do_mma = [&]<uint32_t slot, typename OA>(OA& oaccu_dst) {
        auto run = [&](auto& v) {
            if constexpr(kIsFirstIter) hk::mma_ABt(oaccu_dst, v, p_mfma);
            else hk::mma_ABt(oaccu_dst, v, p_mfma, oaccu_dst);
        };
        if constexpr(slot == 0u) run(pv_v_0);
        else if constexpr(slot == 1u) run(pv_v_1);
        else if constexpr(slot == 2u) run(pv_v_2);
        else run(pv_v_3);
    };

    if constexpr(kDoRescale)
    {
        opus::static_for<2>([&](auto s) {
            pk_mul_pair(rescale, opus::number<k_o_begin + s.value * 4u + 0u>{});
            pk_mul_pair(rescale, opus::number<k_o_begin + s.value * 4u + 2u>{});
        });
    }

    // Prologue: preload S_0,S_1,S_2 (6 ds_read_b64) into slots 0,1,2.
    load_S.template operator()<0u, 0u>();
    load_S.template operator()<1u, 1u>();
    load_S.template operator()<2u, 2u>();

    opus::static_for<num_pv_iter>([&](auto i) {
        constexpr uint32_t iter            = i.value;
        constexpr bool     has_next        = (iter + 1u) < num_pv_iter;
        constexpr uint32_t next_oaccu_base = k_o_begin + (iter + 1u) * 8u;
        constexpr uint32_t j_lo            = 2u * iter;      // flat mfma idx (a)
        constexpr uint32_t j_hi            = 2u * iter + 1u; // flat mfma idx (b)
        // 4-slot round-robin (tile T -> slot T%4). Reading slot j%4 while
        // refilling slot (j+3)%4 keeps the refill ds_read's dst != the mfma's
        // src reg, so it isn't serialized behind the mfma's operand read.
        constexpr uint32_t slot_lo         = j_lo % 4u;
        constexpr uint32_t slot_hi         = j_hi % 4u;

        constexpr uint32_t oaccu_base = k_o_begin + iter * 8u;
        using oaccu_a_r               = hkdart::split_many_t<
            hkdart::type_list<hkdart::range<oaccu_base + 0, oaccu_base + 3>>,
            4>;
        using oaccu_b_r = hkdart::split_many_t<
            hkdart::type_list<hkdart::range<oaccu_base + 4, oaccu_base + 7>>,
            4>;
        hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_a_r> oaccu_a;
        hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_b_r> oaccu_b;

        // ---- mfma_a (flat j_lo, reads slot_lo) ----
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(has_next ? 4 : 2, -1));
        if constexpr(j_lo + 3u < kNumVTiles)
        {
            load_S.template operator()<j_lo + 3u, (j_lo + 3u) % 4u>();
        }
        do_mma.template operator()<slot_lo>(oaccu_a);
        if constexpr(kDoRescale && has_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 0>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 0>{});
        }

        // ---- mfma_b (flat j_hi, reads slot_hi) ----
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(has_next ? 4 : 0, -1));
        if constexpr(j_hi + 3u < kNumVTiles)
        {
            load_S.template operator()<j_hi + 3u, (j_hi + 3u) % 4u>();
        }
        do_mma.template operator()<slot_hi>(oaccu_b);
        if constexpr(kDoRescale && has_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 2>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 2>{});
        }
    });
}

// PV stage selector: picks the kIsFirstIter / kDoRescale instantiation from the
// runtime do_rescale decision the softmax produced.
template <bool kIsFirstIter, typename T>
__device__ __forceinline__ void
hk_mla_v40_pv_stage(KvManager8to16bitsV1<T>& kv_manager, const uintptr_t p_lds_v,
                    const float rescale, const bool do_rescale)
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
