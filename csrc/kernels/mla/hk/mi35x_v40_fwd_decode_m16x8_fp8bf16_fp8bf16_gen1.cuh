// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "hk_mla_softmax.cuh"
#include "hk_mla_v40_buffer_managers_gen1.cuh"
#include "mla.h"
#include <assert.h>
#include <limits>
#include <optional>

using namespace hk_mla;

// Toggle the slim dispatch ladder (fewer mla_main instantiations, always
// boundary-checked prefetch). Comment out to fall back to the full ladder.
#define MLA_SLIM_DISPATCH 1

// Warp-index group. The 8 warps pair onto 4 SIMDs as (w, w+4) -- i.e. SIMD
// pairs are {0,4},{1,5},{2,6},{3,7}. To stagger each SIMD's two warps (one
// PV-at-end, one PV-deferred) we split by HALF, not parity. The DEFERRED path
// goes to the Lo half (warps 0-3, all NoPE) so the RoPE owners' buffer_load_lds
// never lands in a deferred PV's vmcnt window; the Hi half (4-7, incl RoPE 5,7)
// keeps PV at call end. See kPvAtEnd for the path mapping.
enum class WarpType : uint8_t
{
    LoNoPEWarp, // warps 0-3: PV-deferred path, all NoPE
    HiNoPEWarp, // warps 4,6: PV-at-end path, NoPE
    HiRoPEWarp  // warps 5,7: PV-at-end path, RoPE owner
};

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
    // come from LDS). = 2 * (# NoPE chunks in VGPR). Single knob for the
    // incremental NoPE->VGPR migration: 10 (0:320), 12 (0:384), 14 (all NoPE).
    static constexpr uint32_t kQkGemmTiles = 10;

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
    using p_mfma_t =
        hk::art<mfma_ab_t, T::kTileM, T::kBlockN, hk::row_l, hk::rt_16x32_s, p_mfma_ranges>;
    using oaccu_t = hk::art<comp_t, T::kTileM, T::kVoHeadDim, hk::row_l, hk::rt_16x16_s, o_ranges>;
};

// ---- PV GEMM stage (extracted so even/odd orchestrators share one body) ----
//
// O = P @ V computed as oaccu^T = V^T @ P^T via mma_ABt(oaccu, V, p_mfma).
// V streams as 32 base tiles S_0..S_31 through a 3-deep round-robin
// pv_v_0/pv_v_1/pv_v_2 (S_j -> slot j%3, 6 ds_read_b64 in flight). p_lds_v is
// the V pong (curr for the even path; the prior call's next-pong for the rotated
// odd path). kDoRescale folds the online-softmax oaccu rescale; kIsFirstIter
// inits oaccu fresh (3-arg mma) and never rescales.
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
    typename R::p_mfma_t p_mfma;
    typename R::pv_v_0_t pv_v_0;
    typename R::pv_v_1_t pv_v_1;
    typename R::pv_v_2_t pv_v_2;

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

    // Issue both ds_read_b64 for base tile S_jj into round-robin slot.
    auto load_S = [&]<uint32_t jj, uint32_t slot>() {
        constexpr uint32_t base =
            (slot == 0u) ? k_v0_begin : (slot == 1u) ? k_k0_begin : k_k1_begin;
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
        else run(pv_v_2);
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
        constexpr uint32_t slot_lo         = j_lo % 3u;
        constexpr uint32_t slot_hi         = j_hi % 3u;

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
        do_mma.template operator()<slot_lo>(oaccu_a);
        if constexpr(j_lo + 3u < kNumVTiles)
        {
            load_S.template operator()<j_lo + 3u, slot_lo>();
        }
        if constexpr(kDoRescale && has_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 0>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 0>{});
        }

        // ---- mfma_b (flat j_hi, reads slot_hi) ----
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(has_next ? 4 : 0, -1));
        do_mma.template operator()<slot_hi>(oaccu_b);
        if constexpr(kDoRescale && has_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 2>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 2>{});
        }
        if constexpr(j_hi + 3u < kNumVTiles)
        {
            load_S.template operator()<j_hi + 3u, slot_hi>();
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

// V4.0 mi35x m16x8 decode kernel: separate FP8 NOPE + BF16 ROPE buffers for
// both Q and KV. End-to-end body (Phases 4a..4g) in place: prologue (Q load +
// first KV tile) -> per-warp dispatch ladder over mla_main (QK GEMM + softmax
// + PV GEMM + epilogue, with online-softmax rescale across K-tile iters).
#if defined(__gfx950__)
template <WarpType kWarpType, typename T>
__device__ void mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1_impl(HkMlaV40DecodeFwdParams<T>
                                                                          params,
                                                                          const uint32_t warp_idx)
{
    using q_nope_t  = T::q_nope_t;
    using q_rope_t  = T::q_rope_t;
    using kv_nope_t = T::kv_nope_t;
    using kv_rope_t = T::kv_rope_t;
    using out_t     = T::out_t;
    using comp_t    = float;
    using split_t   = float; // format of temp split output and lse.
    // All MFMA operands live in bf16 after the QManager/KvManager cvt step.
    using mfma_ab_t = hk::bf16;

    using G = hk::group<T::kNumWarps>;

    // Compile-time warp type (chosen by the __global__ wrapper's entry divergence).
    constexpr bool kIsRopeWarp = (kWarpType == WarpType::HiRoPEWarp);
    // kPvAtEnd: Hi group (warps 4-7, incl RoPE owners 5,7) keeps PV at call end;
    // Lo group (warps 0-3, all NoPE) defers PV by one tile so its prefetch+SALU
    // overlaps the Hi sibling's PV. The deferred path is given to Lo (NoPE-only)
    // so the RoPE warps' buffer_load_lds never collides with a deferred PV in the
    // prefetch's vmcnt window. SIMD pairs (w, w+4) still get one of each path.
    constexpr bool kPvAtEnd = (kWarpType != WarpType::LoNoPEWarp);
    (void)kPvAtEnd;

    constexpr comp_t log2e = 1.4426950408889634;

    const int32_t worker_idx     = blockIdx.x;
    const int32_t work_start_idx = __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx]);
    const int32_t work_end_idx =
        __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx + 1]);
    if(work_start_idx >= work_end_idx)
    {
        return;
    }

    // ---- VGPR layout (per-lane) ----
    // Compiler scratch constrained to v0..v43 (budget=44 on the __global__).
    // Full-Q-VGPR map (chunks 0-4 populated this step):
    //   255:128 oaccu
    //   127: 64 q_vgpr (full Q; v64..v103 = chunks 0-4, v104..v111 = q_lds scratch)
    //    63: 56 p_comp (60:63 = HI half -> PV V tile v_0); 56:59 p_mfma overlay
    //    55: 52 k_0    (also PV V tile v_1)
    //    51: 48 k_1
    //    47: 44 k_2
    // Register layout single source of truth (see HkMlaV40Regs at file scope).
    // Re-alias the constants/ranges locally so the body below keeps its existing
    // k_* / *_ranges names; both orchestrators + every stage bind the same regs.
    using R = HkMlaV40Regs<T>;
    constexpr uint32_t mfma_tile_sz   = R::mfma_tile_sz;
    constexpr uint32_t k_o_begin      = R::k_o_begin;
    constexpr uint32_t k_o_end        = R::k_o_end;
    constexpr uint32_t k_p_comp_begin = R::k_p_comp_begin;
    constexpr uint32_t k_p_comp_end   = R::k_p_comp_end;
    constexpr uint32_t k_p_mfma_begin = R::k_p_mfma_begin;
    constexpr uint32_t k_p_mfma_end   = R::k_p_mfma_end;
    constexpr uint32_t k_v0_begin     = R::k_v0_begin;
    constexpr uint32_t k_v0_end       = R::k_v0_end;
    constexpr uint32_t k_q_vgpr_begin = R::k_q_vgpr_begin;
    constexpr uint32_t k_q_vgpr_end   = R::k_q_vgpr_end;
    constexpr uint32_t k_k0_begin     = R::k_k0_begin;
    constexpr uint32_t k_k1_begin     = R::k_k1_begin;
    constexpr uint32_t k_k2_begin     = R::k_k2_begin;
    constexpr uint32_t k_q_lds_1_begin = R::k_q_lds_1_begin;
    constexpr uint32_t k_q_lds_0_begin = R::k_q_lds_0_begin;
    constexpr uint32_t k_q_lds_begin   = R::k_q_lds_begin;
    (void)mfma_tile_sz;
    (void)k_o_end;
    (void)k_p_comp_end;
    (void)k_p_mfma_end;
    (void)k_v0_end;
    (void)k_q_vgpr_end;
    (void)k_k2_begin;
    (void)k_q_lds_1_begin;
    (void)k_q_lds_0_begin;

    using q_vgpr_ranges     = typename R::q_vgpr_ranges;
    using p_comp_ranges     = typename R::p_comp_ranges;
    using p_comp_lo_ranges  = typename R::p_comp_lo_ranges;
    using p_comp_hi_ranges  = typename R::p_comp_hi_ranges;
    using kv_top_ranges     = typename R::kv_top_ranges;
    using kv_bot_ranges     = typename R::kv_bot_ranges;
    using kv_alt_top_ranges = typename R::kv_alt_top_ranges;
    using p_mfma_ranges     = typename R::p_mfma_ranges;
    using o_ranges          = typename R::o_ranges;
    using pv_v_0_ranges     = typename R::pv_v_0_ranges;
    using pv_v_1_ranges     = typename R::pv_v_1_ranges;
    using pv_v_2_ranges     = typename R::pv_v_2_ranges;
    using q_lds_ranges      = typename R::q_lds_ranges;

    hkdart::clobber<q_vgpr_ranges>();
    hkdart::clobber<p_comp_ranges>();
    hkdart::clobber<p_mfma_ranges>();
    hkdart::clobber<o_ranges>();
    hkdart::clobber<q_lds_ranges>();

    // ---- Managers ----
    QManager8to16bitsV1<T> q_manager;
    KvManager8to16bitsV1<T> kv_manager;
    OManager16bitsV4Gen1Swizzle<T, out_t> o_manager;
    OManager32bitsV4Gen1Swizzle<T, split_t> split_o_manager;

    // ---- art tile declarations (bound to the shared register ranges) ----
    typename R::q_vgpr_t q_vgpr;
    typename R::p_comp_t p_comp;
    typename R::p_comp_lo_t p_comp_lo;
    typename R::p_comp_hi_t p_comp_hi;
    typename R::k_0_t k_0;
    typename R::k_1_t k_1;
    typename R::k_2_t k_2;
    typename R::pv_v_0_t pv_v_0;
    typename R::pv_v_1_t pv_v_1;
    typename R::pv_v_2_t pv_v_2;
    typename R::p_mfma_t p_mfma;
    typename R::oaccu_t oaccu;

    // ---- Runtime constants ----
    const uint32_t lane_idx = opus::lane_id();

    // Causal mask: compute per-warp kv_end offset for MTP.
    // num_wave_group = qseqlen = kBlockM / num_qheads
    // waves_per_head = num_qheads / kTileM
    // causal_offset = num_wave_group - 1 - (warp_idx / waves_per_head)
    const int32_t log2_num_qheads = __builtin_amdgcn_readfirstlane(params.log2_num_qheads);
    const int32_t num_qheads      = 1 << log2_num_qheads;

    // Per-lane attention sink logit. Loaded once at kernel entry: it depends
    // only on (warp_idx, lane_idx), not on work_idx, so it lives in a VGPR
    // for the kernel's lifetime. When p_attn_sink is null, substitute -inf
    // so exp(sink - row_max) = 0 -> the epilogue's row_sum_e += sink_term
    // becomes a no-op. num_qheads is a power of 2 in {16,32,64,128} (see
    // outer wrapper check).
    const uint32_t head_idx =
        (warp_idx * 16u + (lane_idx & 15u)) & (static_cast<uint32_t>(num_qheads) - 1u);
    const float attn_sink = (params.p_attn_sink == nullptr)
                                ? -std::numeric_limits<float>::infinity()
                                : params.p_attn_sink[head_idx];

    const int32_t num_wave_group      = T::kBlockM >> log2_num_qheads; // qseqlen
    const int32_t log2_waves_per_head = log2_num_qheads - 4;           // log2(kTileM) = 4
    const int32_t qpos_off_from_last  = num_wave_group - 1 - (warp_idx >> log2_waves_per_head);

    // ---- LDS layout ----
    //
    // p_lds_kv_curr/   : 32 KB each (32 rows * 512 bf16 cols, 2 pongs).
    //  p_lds_kv_next     Placed FIRST so they cover the +0 LDS base.
    // O bounce         : overlays p_lds_kv_next (the next pong is DEAD on the
    //                    global last iter, where the epilogue runs, since the
    //                    swap is a no-op). Per-warp strides differ between
    //                    QManager (8 KB) and OManager V3 (2112 B bf16 /
    //                    4352 B fp32), so placing the O bounce inside p_lds_q
    //                    creates cross-warp aliasing with the next work_idx's
    //                    load_q -- racy when a fast warp's load_q lands while
    //                    a slow warp's epilogue is still in flight. Overlaying
    //                    KV-next instead keeps the O bounce in a region whose
    //                    next consumer (next work_idx's KV prologue) writes to
    //                    p_lds_kv_curr, not next.
    // p_lds_q          : 64 KB - QManager region. Placed AFTER both KV pongs +
    //                    max(O, KV) so the O bounce never overlaps with Q,
    //                    and so warp 0's Phase-1 staging (at p_lds_q + 0)
    //                    starts well above 0 in m0 -- this lets
    //                    p1_vmem_to_staging_chunk pre-subtract up to 192 B
    //                    (kColInRecord = 0/64/128/192) from the LDS dst without
    //                    m0 underflowing mod 2^32.
    //
    // Total (occupancy=1): KvLds + max(KvLds, O) + 64 KB Q.
    extern __shared__ int32_t p_lds[];

    // opus::max is device-only / non-constexpr; use inline ternary in constexpr
    // contexts.
    constexpr uint32_t kSzLdsQ  = q_manager.get_lds_size_in_byte();
    constexpr uint32_t kSzLdsKv = kv_manager.get_lds_size_in_byte();
    constexpr uint32_t kSzLdsO =
        (o_manager.get_lds_size_in_byte() > split_o_manager.get_lds_size_in_byte())
            ? o_manager.get_lds_size_in_byte()
            : split_o_manager.get_lds_size_in_byte();

    static_assert(kSzLdsQ + kSzLdsKv + (kSzLdsO > kSzLdsKv ? kSzLdsO : kSzLdsKv) <= 160u * 1024u,
                  "V4.0 LDS budget exceeds 160 KB at kOccupancy=1.");
    // QManager pre-subtracts up to kLdsHeadPadBytes from p_lds_q in
    // p1_vmem_to_staging_chunk. Placing Q after both KV pongs gives that
    // subtraction enough headroom (m0 lands in KV-pong region, still valid LDS).
    static_assert(kSzLdsKv + (kSzLdsO > kSzLdsKv ? kSzLdsO : kSzLdsKv) >=
                      QManager8to16bitsV1<T>::kLdsHeadPadBytes,
                  "KV pongs must precede Q LDS with enough bytes to absorb the "
                  "QManager P1 pre-subtract.");

    uintptr_t p_lds_kv_curr = reinterpret_cast<uintptr_t>(p_lds);
    uintptr_t p_lds_kv_next = p_lds_kv_curr + kSzLdsKv;
    const uintptr_t p_lds_q = p_lds_kv_next + (kSzLdsO > kSzLdsKv ? kSzLdsO : kSzLdsKv);

    // ---- Work loop ----
    // Phase 4b is in place: per work item, read work_info, resolve kv extents,
    // load Q (vmem -> VGPR + bf16 LDS), and prefetch+cvt+store the first KV
    // tile into the curr pong. The mla_main lambda + dispatch ladder still TODO
    // (Phases 4c-4f); kernel still hits assert(false) at the bottom of the loop.
    const uint32_t kv_ld_row_base_idx = kv_manager.get_kv_ld_row_base_idx(warp_idx);

    for(int32_t work_idx = work_start_idx; work_idx < work_end_idx; ++work_idx)
    {
        const int32_t batch_idx = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 0]);
        const int32_t partial_qo_loc = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 1]);
        const int32_t qo_start = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 2]);
        const int32_t qo_end = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 3]);
        const int32_t kv_start_page = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 4]);
        const int32_t kv_end_page = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 5]);
        // kv_offset == 0 iff this work item ends at the batch tail (kPageSize > 1).
        const int32_t kv_offset = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 6]);

        // "Last split of this batch" -- the planner sets
        // kv_offset = curr_kv_end - work_info.kv_end (metadata/v1_2_device.cuh
        // L202/L214), so kv_offset == 0 iff this split's kv_end coincides
        // with the batch tail. Used by the epilogue sink fold: only the
        // LAST split inflates row_sum_e with the sink term, so the reducer
        // routes the sink contribution into the global denominator exactly
        // once. Last-vs-first is mathematically equivalent (reducer combines
        // lses commutatively); last-split is cheaper -- no extra kv_indptr
        // load (kv_offset is already in scope above).
        const bool is_last_split = (kv_offset == 0);

        // Convert work_info page bounds to TOKEN space. When kPageSize == 1
        // pages == tokens. When kPageSize > 1 and this is the batch tail
        // (kv_offset == 0), clip the last page with kv_last_page_lens[batch].
        // The (kPageSize == 1) check folds at compile time so the load is
        // dead-code-eliminated for kPageSize == 1.
        const int32_t kv_start = kv_start_page * T::kPageSize;
        int32_t kv_end;
        if((T::kPageSize == 1) || (kv_offset != 0))
        {
            kv_end = kv_end_page * T::kPageSize;
        }
        else
        {
            const int32_t last_page_len =
                __builtin_amdgcn_readfirstlane(params.p_kv_last_page_lens[batch_idx]);
            kv_end = (kv_end_page - 1) * T::kPageSize + last_page_len;
        }
        // Per-warp causal mask: qpos i sees kv_end - max(0, qpos_off_from_last - kv_offset).
        const int32_t causal_offset = opus::max(qpos_off_from_last - kv_offset, 0);
        const int32_t kv_end_eff    = kv_end - causal_offset;
        const int32_t kv_len        = kv_end - kv_start;
        const int32_t kv_len_eff    = kv_end_eff - kv_start;

        // Online-softmax running stats. Each warp owns one (16-row) M-tile; the
        // values are lane-private (each lane holds the stats for its 1/64th
        // share of the tile, established by the warp_reduce inside softmax_p0).
        comp_t row_max;
        comp_t row_sum_e;
        // rescale / do_rescale are produced by each tile's softmax and consumed
        // by its PV. Hoisted to work-item scope so the ODD (rotated) path can
        // read the PREVIOUS tile's values in the next call's deferred PV; the
        // softmax assigns (not declares) them, so the deferred PV reads the prior
        // tile's values before the current softmax overwrites them. Even path:
        // set + used in the same call, so the hoist is inert.
        comp_t rescale  = 1.0f;
        bool do_rescale = false;

        // Helper: resolve the physical KV row for the 32-row tile that begins
        // at tile_start. Returns -1 if the tile is entirely OOB.
        auto resolve_row_kv_ld = [&](const int32_t tile_start) -> int32_t {
            const int32_t tile_end = tile_start + T::kBlockN;
            int32_t row_kv_ld;
            if(tile_end <= kv_end)
            {
                row_kv_ld = get_kv_ld_row<false, T::kPageSize>(
                    params.p_kv_indices, kv_ld_row_base_idx, tile_start, tile_end);
            }
            else if(tile_start < kv_end)
            {
                row_kv_ld = get_kv_ld_row<true, T::kPageSize>(
                    params.p_kv_indices, kv_ld_row_base_idx, tile_start, kv_end);
            }
            else
            {
                row_kv_ld = -1;
            }
            return row_kv_ld;
        };

        // Tile 0's KV row goes to the prologue; tile 1's seed row goes to the
        // first lambda call's prefetch. `row_kv_ld_next_next` is a one-deep
        // carry: each lambda call snapshots it for its prefetch and updates it
        // for the call after (matching V32's per-warp dispatch pattern).
        const int32_t row_kv_ld_first = resolve_row_kv_ld(kv_start);
        int32_t row_kv_ld_next_next =
            (kv_len > T::kBlockN) ? resolve_row_kv_ld(kv_start + T::kBlockN) : -1;

        // Load Q: Q[:, 0:256] -> VGPR pinned at k_q_vgpr_begin (32 vgprs/lane).
        //         Q[:, 256:512] -> bf16 final LDS region inside p_lds_q.
        // Q rope/nope buffers are separate tensors in V4.0.
        // Fold the softmax temperature AND the natural->log2 conversion into Q
        // once here, so the per-KV-tile softmax drops both its sm_scale multiply
        // (softmax_scale_p -> softmax_mask_p) and its log2e multiply
        // (softmax_p1 -> softmax_p1_prescaled). Scores then arrive in log2 units.
        const float q_scale_log2 = params.softmax_scale * static_cast<float>(log2e);
        q_manager.template load_q<k_q_vgpr_begin, R::kQkGemmTiles / 2u>(params.p_query,
                                                  params.p_query_rope,
                                                  num_qheads,
                                                  warp_idx,
                                                  qo_start,
                                                  p_lds_q,
                                                  q_scale_log2);
        __builtin_amdgcn_sched_barrier(0);

        // Prologue: prefetch + cvt+store the first KV tile into the curr pong.
        // kCheckBoundary is true when the tile straddles the batch tail.
        if(kv_len < T::kBlockN)
        {
            kv_manager.template async_load_k<true, kIsRopeWarp>(p_lds_kv_curr,
                                                   warp_idx,
                                                   params.p_kv_buffer,
                                                   params.p_kv_buffer_rope,
                                                   row_kv_ld_first);
        }
        else
        {
            kv_manager.template async_load_k<false, kIsRopeWarp>(p_lds_kv_curr,
                                                    warp_idx,
                                                    params.p_kv_buffer,
                                                    params.p_kv_buffer_rope,
                                                    row_kv_ld_first);
        }

        // ---- mla_main lambda (Phase 4g) ----
        //
        // One K-tile iter. Templates:
        //   kIsFirstIter      : this is the warp's first compute iter (oaccu
        //                       gets initialized by PV's 3-arg mfma, no
        //                       rescale needed against prior row_max/oaccu).
        //   kSkipCompute      : warp is idle on this tile (e.g., causal-masked
        //                       trailing iter); only barriers + KV cooperative
        //                       work run. Implies !kIsFirstIter.
        //   kEpilogueType     : None (continue) / OutputFinal / OutputSplit.
        //   kCheckBoundaryNext: the NEXT tile may be OOB (partial last tile);
        //                       prefetch uses kCheckBoundary=true.
        //
        // Derived: kDoEpilogue = (kEpilogueType != None);
        //          kIsGlobalLast = kSkipCompute || kDoEpilogue.
        // kIsGlobalLast means no next tile to load -- skip prefetch, wait, swap.
        auto mla_main = [&]<bool kIsFirstIter,
                            bool kSkipCompute,
                            PvGemmEpilogueType kEpilogueType,
                            bool kCheckBoundaryNext>(const int32_t kv_tile_start,
                                                     const int32_t kv_tile_end) {
            constexpr bool kDoEpilogue   = (kEpilogueType != PvGemmEpilogueType::None);
            constexpr bool kIsGlobalLast = kSkipCompute || kDoEpilogue;
            (void)kv_tile_end;

            // ODD (rotated) path: PV is deferred one tile so this call's
            // prefetch+SALU overlaps the EVEN sibling wave's PV on the shared
            // SIMD. A deferred PV (of the PREVIOUS tile) runs at call start for
            // every call that has a not-yet-PV'd prior tile -- all but the
            // warp's first compute call and the fully-idle skip call.
            constexpr bool kHasPv = (!kPvAtEnd) && (!kIsFirstIter) &&
                                    ((!kSkipCompute) || kDoEpilogue);
            // The deferred first PV uses the accumulate path (3-arg init is
            // even-only), so the odd path zeroes oaccu once on its first compute
            // call instead of relying on PV's init. row_max/row_sum_e are still
            // initialised by the kIsFirstIter softmax below, shared with even.
            if constexpr((!kPvAtEnd) && kIsFirstIter)
            {
                hk::zero(oaccu);
            }

            static_assert((kSkipCompute == false) || (kIsFirstIter == false),
                          "A skipped iter cannot be the warp's first compute iter.");
            static_assert(
                (kIsGlobalLast == false) || (kCheckBoundaryNext == false),
                "kIsGlobalLast == true means no next tile, so kCheckBoundaryNext must be false.");

            // Snapshot next-tile KV row (set by prior call or prologue).
            int32_t row_kv_ld_next = 0;
            if constexpr(kIsGlobalLast == false)
            {
                row_kv_ld_next = row_kv_ld_next_next;
                // Deferred-PV (Lo) group: hoist the page-table resolve for the
                // tile AFTER next to call start (only needs kv_tile_start, no QK
                // result) so its buffer_load overlaps this call's deferred PV +
                // QK and the NEXT call's prefetch reads the row without a stall.
                // PV-at-end (Hi) group keeps it late (below).
                if constexpr(!kPvAtEnd)
                {
                    row_kv_ld_next_next = resolve_row_kv_ld(kv_tile_start + 2 * T::kBlockN);
                }
            }

            // ---- Phase A: prefetch NEXT tile into the next-pong ----
            // 2 halves (kColOffset 0 + 256) per tile. Carriers live in VGPRs
            // until the wait+cvt+store sequence below; the gap in between
            // hides vmcnt latency under QK MFMAs.
            constexpr uint32_t kTileCols = 256u;
            typename KvManager8to16bitsV1<T>::KvTilePrefetch p0, p1;

            // Issue both tiles' NoPE prefetches before the barrier so their vmem
            // load latency overlaps the barrier wait. NoPE is VGPR-landing only
            // (no LDS write), so it's safe ahead of the barrier that protects
            // p_lds_kv_next. The RoPE halves (tile 1, waves 5,7) DMA straight to
            // LDS, so they stay AFTER the barrier (issued in the QK body below).
            //
            // Per-wave vmem count after this block:
            //   waves 0-4,6: 4 loads (2 NoPE dwordx4 + 2 scale ubyte, both tiles)
            //   waves 5,7:   2 loads (tile-0 NoPE only; tile-1 is their RoPE half)
            // so the post-barrier wait below is wave-dependent.
            if constexpr((kSkipCompute == false) && (kIsGlobalLast == false))
            {
                kv_manager.template prefetch_kv_nope<0u, 0u, kCheckBoundaryNext, kIsRopeWarp>(
                    warp_idx, params.p_kv_buffer, row_kv_ld_next, p0);
                kv_manager.template prefetch_kv_nope<0u, kTileCols, kCheckBoundaryNext, kIsRopeWarp>(
                    warp_idx, params.p_kv_buffer, row_kv_ld_next, p1);
            }

            // ---- ODD deferred PV (of the PREVIOUS tile) ----
            // Issued AFTER this call's prefetch so the prefetch+SALU lands while
            // the even sibling wave is mid-PV. Reads the previous tile's V from
            // p_lds_kv_next (left there by the prior call's swap, still alive --
            // this call's cvt+store has not run yet) and the previous tile's
            // p_mfma / rescale / do_rescale (pinned regs + hoisted scalars, not
            // yet overwritten by this call's QK/softmax). Accumulate path only.
            if constexpr(kHasPv)
            {
                hk_mla_v40_pv_stage<false, T>(kv_manager, p_lds_kv_next, rescale, do_rescale);
            }

            // Drain the NoPE prefetches enough to leave only this iter's own
            // prologue loads pending, then cross-warp barrier so KV LDS
            // sub-blocks are visible to QK reads. RoPE-owner waves issued 2
            // fewer loads, so they wait on vmcnt(2) vs vmcnt(4).
            if constexpr(kIsRopeWarp)
            {
                __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/2));
            }
            else
            {
                __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/4));
            }
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // ---- QK GEMM (simple generic full-drain; correctness-first) ----
            // One mfma-pair per global col-tile gct in [0,16): K both row-halves
            // from KV-LDS, Q from q_vgpr (gct < kQkGemmTiles) or from Q-LDS
            // (gct >= kQkGemmTiles). Full lgkmcnt(0) drain before each pair -> no
            // round-robin / wait tuning. kQkGemmTiles is the only migration knob.
            constexpr uint32_t kQkGemmTiles = R::kQkGemmTiles;
            if constexpr(kSkipCompute == false)
            {
                constexpr uint32_t kBK             = T::kBlockK;
                constexpr uint32_t kNumColTilesAll = T::kQkHeadDim / T::kBlockK; // 16
                using q_lds_range_0 = hkdart::split_many_t<
                    hkdart::type_list<hkdart::range<k_q_lds_begin, k_q_lds_begin + 3u>>, 4>;
                hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_lds_range_0>
                    q_lds_0;

                opus::static_for<kNumColTilesAll>([&](auto g_) {
                    constexpr uint32_t gct = g_.value;
                    kv_manager.template load_k_to_gpr<0u, gct * kBK>(k_0, p_lds_kv_curr);
                    kv_manager.template load_k_to_gpr<16u, gct * kBK>(k_1, p_lds_kv_curr);
                    if constexpr(gct < kQkGemmTiles)
                    {
                        constexpr uint32_t q_base = k_q_vgpr_begin + gct * 4u;
                        using q_range = hkdart::split_many_t<
                            hkdart::type_list<hkdart::range<q_base, q_base + 3u>>, 4>;
                        hk::art<mfma_ab_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_range>
                            q_tile;
                        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, -1));
                        if constexpr(gct == 0u)
                        {
                            hk::mma_ABt(p_comp_lo, k_0, q_tile);
                            hk::mma_ABt(p_comp_hi, k_1, q_tile);
                        }
                        else
                        {
                            hk::mma_ABt(p_comp_lo, k_0, q_tile, p_comp_lo);
                            hk::mma_ABt(p_comp_hi, k_1, q_tile, p_comp_hi);
                        }
                    }
                    else
                    {
                        q_manager.template load_q_lds_to_gpr<gct - 8u>(q_lds_0, p_lds_q, warp_idx);
                        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, -1));
                        hk::mma_ABt(p_comp_lo, k_0, q_lds_0, p_comp_lo);
                        hk::mma_ABt(p_comp_hi, k_1, q_lds_0, p_comp_hi);
                    }
                });
            }

            // ---- Phase B+C: wait + cvt + store NEXT tile to LDS ----
            // Sequenced after QK so the QK ds_reads from p_lds_kv_curr aren't
            // delayed by the cvt+store traffic on p_lds_kv_next.
            if constexpr(kIsGlobalLast == false)
            {
                constexpr uint32_t kTileCols = 256u;
                if constexpr(kSkipCompute)
                {
                    // No QK GEMM ran -> prefetches weren't issued inline.
                    // Full prefetch + wait + cvt + store via async_load_k.
                    kv_manager.template async_load_k<kCheckBoundaryNext, kIsRopeWarp>(p_lds_kv_next,
                                                                         warp_idx,
                                                                         params.p_kv_buffer,
                                                                         params.p_kv_buffer_rope,
                                                                         row_kv_ld_next);
                }
                else
                {
                    // RoPE half of NEXT tile 1 (waves 5,7 only) DMAs straight
                    // into p_lds_kv_next. Moved here (after QK GEMM) from the
                    // mid-QK Pair P0 slot. Must be issued before the tile-1
                    // wait below so its vmem load is counted. NoPE halves were
                    // issued pre-barrier; this is vmcnt, the QK ds_reads are
                    // lgkmcnt -- different counters.
                    kv_manager.template prefetch_kv_rope<0u, kTileCols, kCheckBoundaryNext, kIsRopeWarp>(
                        p_lds_kv_next, warp_idx, params.p_kv_buffer_rope, row_kv_ld_next);

                    // Prefetches already issued mid-QK-body. Just drain +
                    // cvt + store. 4 vmem ops in flight; first wait drains
                    // tile-0 (leave tile-1's 2 in flight); second drains tile-1.
                    hk::u32x4 dw;
                    kv_manager.template wait_kv_loads<0u, 0u, kIsRopeWarp, /*kVmCnt=*/2>(warp_idx);
                    const float scale_f0 = kv_manager.kv_tile_scale_f(p0);
                    kv_manager.template cvt_kv_tile_step<0>(dw, p0, scale_f0);
                    kv_manager.template cvt_kv_tile_step<1>(dw, p0, scale_f0);
                    kv_manager.template store_kv_tile_step<0u, 0u, 0, kIsRopeWarp>(p_lds_kv_next, warp_idx, dw);
                    kv_manager.template cvt_kv_tile_step<2>(dw, p0, scale_f0);
                    kv_manager.template cvt_kv_tile_step<3>(dw, p0, scale_f0);
                    kv_manager.template store_kv_tile_step<0u, 0u, 1, kIsRopeWarp>(p_lds_kv_next, warp_idx, dw);

                    // Tile-1 is the RoPE half for waves 5,7 -- they have no NoPE
                    // data in p1 (prefetch_kv_nope was a no-op) and their
                    // store_kv_tile_step already skips, so skip the scale+cvts
                    // too (they'd compute discarded garbage from uninit p1).
                    if constexpr(!kIsRopeWarp)
                    {
                        kv_manager.template wait_kv_loads<0u, kTileCols, kIsRopeWarp, /*kVmCnt=*/0>(warp_idx);

                        const float scale_f1 = kv_manager.kv_tile_scale_f(p1);
                        kv_manager.template cvt_kv_tile_step<0>(dw, p1, scale_f1);
                        kv_manager.template cvt_kv_tile_step<1>(dw, p1, scale_f1);
                        kv_manager.template store_kv_tile_step<0u, kTileCols, 0, kIsRopeWarp>(
                            p_lds_kv_next, warp_idx, dw);
                        kv_manager.template cvt_kv_tile_step<2>(dw, p1, scale_f1);
                        kv_manager.template cvt_kv_tile_step<3>(dw, p1, scale_f1);
                        kv_manager.template store_kv_tile_step<0u, kTileCols, 1, kIsRopeWarp>(
                            p_lds_kv_next, warp_idx, dw);
                    }
                }
            }

            __builtin_amdgcn_s_setprio(2);

            // ---- Update row_kv_ld_next_next for the call AFTER this one ----
            // PV-at-end (Hi) group only: resolve here, just after the QK phase;
            // its buffer_load overlaps softmax + the end-PV before the next
            // call's prefetch. The deferred (Lo) group already resolved at call
            // start (above). resolve_row_kv_ld returns -1 past the global end --
            // the subsequent boundary-checked prefetch suppresses that load.
            if constexpr((kIsGlobalLast == false) && kPvAtEnd)
            {
                row_kv_ld_next_next = resolve_row_kv_ld(kv_tile_start + 2 * T::kBlockN);
            }

            // ---- Softmax + fp32->bf16 pack ----
            //
            // p_comp is 8 fp32 lanes (kBlockN=32 N-cols x kTileM=16 rows / 64
            // lanes = 8 elems/lane), laid out per softmax_scale_p_8: lane's
            // col_0 group covers vgprs +0..+3 (N-cols [col_0_idx*4, +4)) and
            // col_1 group covers +4..+7 (N-cols [col_0_idx*4+16, +20)).
            const uint32_t col_0_idx = lane_idx >> 4;
            comp_t local_max{};
            rescale = 1.0f;
            // Wave-uniform: does the running max move enough this tile to
            // require rescaling the prior oaccu / row_sum_e? Decided by ballot
            // so the whole wave agrees (oaccu rescale is a per-wave op, but
            // each lane owns different rows). Stays false on kIsFirstIter (no
            // prior oaccu) and whenever every active lane's new max is within
            // T::kRescaleThreshold of the stale max -- in which case row_max is
            // kept stale and the rescale multiplies (all == 1) are skipped.
            do_rescale = false;
            if constexpr(kSkipCompute == false)
            {
                // Q was pre-scaled by sm_scale*log2e in load_q, so only mask
                // OOB columns here (no per-tile multiply).
                const uint32_t kv_tile_start_u = static_cast<uint32_t>(kv_tile_start);
                if((kv_tile_start_u + T::kBlockN) > static_cast<uint32_t>(kv_end_eff))
                {
                    softmax_mask_p<true, k_p_comp_begin>(col_0_idx * 4u + kv_tile_start_u,
                                                         static_cast<uint32_t>(kv_end_eff));
                }
                else
                {
                    softmax_mask_p<false, k_p_comp_begin>(col_0_idx * 4u + kv_tile_start_u,
                                                          static_cast<uint32_t>(kv_end_eff));
                }

                // Row-wise max across 8 p_comp vgprs, then across the 4-lane
                // M-group via warp_reduce (matches softmax_p0's reduction).
                local_max = max_8<k_p_comp_begin, comp_t>();
                {
                    constexpr int32_t reduce_range = opus::get_warp_size();
                    constexpr int32_t stop_stride  = opus::get_warp_size() / 4 - 1;
                    local_max                      = warp_reduce<aiter::MaxFunctor,
                                                                 decltype(local_max),
                                                                 reduce_range,
                                                                 stop_stride>(local_max);
                }
                if constexpr(kIsFirstIter)
                {
                    row_max = local_max;
                    rescale = 1.0f;
                }
                else
                {
                    // Lane-private: would this lane's rows need a rescale?
                    const bool lane_needs =
                        (local_max - row_max) > static_cast<comp_t>(T::kRescaleThreshold);
                    // Promote to a wave-uniform decision: rescale iff ANY active
                    // lane needs it (ballot != 0). When no lane needs it, keep
                    // row_max stale so exp(p_comp - row_max) accumulates into the
                    // existing oaccu reference (rescale stays 1.0, mults skipped).
                    do_rescale = (__builtin_amdgcn_ballot_w64(lane_needs) != 0ull);
                    if(do_rescale)
                    {
                        const comp_t new_row_max = opus::max(local_max, row_max);
                        // row_max is already in log2 units (Q pre-scaled), so no
                        // * log2e here.
                        rescale  = __builtin_amdgcn_exp2f(row_max - new_row_max);
                        row_max  = new_row_max;
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(1);
                __builtin_amdgcn_sched_barrier(0);

                // exp + sum + warp_reduce(add) -> row_sum_e. Updates p_comp in
                // place to hold exp(p_comp - row_max). rescale==1.0 when the
                // running max was kept stale, so the prior row_sum_e carries
                // forward unscaled.
                softmax_p1_prescaled<kIsFirstIter, k_p_comp_begin>(&row_sum_e, row_max, rescale);

                if constexpr(kPvAtEnd == false)
                {
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_setprio(0);
                    __builtin_amdgcn_sched_barrier(0);
                }

                // ---- fp32->bf16 pack (p_comp -> p_mfma overlay) ----
                // 8 fp32 (v120..v127) -> 4 bf16x2 dwords (v120..v123 overlay).
                // Low-to-high pack order is hazard-free: each v_cvt_pk_bf16_f32
                // is atomic (reads sources before writing dst), and no later
                // pack reads a vgpr that an earlier pack overwrote.
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 0, k_p_comp_begin + 0>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 1, k_p_comp_begin + 2>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 2, k_p_comp_begin + 4>();
                pack_2f32_to_bf16_pair_pinned<k_p_mfma_begin + 3, k_p_comp_begin + 6>();
            }

            // ---- oaccu rescale + PV GEMM ----
            //
            // Templated on kDoRescale so the online-softmax oaccu rescale can
            // be elided entirely when the running max did not move this tile
            // (do_rescale==false). kDoRescale==false leaves a clean PV gemm
            // with no v_mul_f32/v_pk_mul_f32 interleaved; kDoRescale==true is
            // the full path. kDoRescale is only ever instantiated true when
            // kIsFirstIter==false (the first iter inits oaccu fresh, never
            // rescales), so kDoRescale implies !kIsFirstIter.
            //
            // PV GEMM: O = P @ V, computed as oaccu^T = V^T @ P^T via
            // mma_ABt(oaccu, V, p_mfma). With kVoHeadDim=512 and kBlockN=32 the V
            // operand is 32 base tiles S_0..S_31 (S_j = V-cols [j*16:(j+1)*16],
            // 4 vgprs = 2 ds_read_b64_tr_b16 each). They stream through a 3-deep
            // round-robin over pv_v_0/pv_v_1/pv_v_2 (S_j -> slot j%3), keeping 6
            // ds_read_b64 (= 3 base tiles) in flight. Flat mfma_j (j=0..31) reads
            // slot(j%3) then issues reload S_{j+3}->slot(j%3) (mfma-then-ds_read,
            // safe WAR order). Steady wait lgkmcnt(4) drains the operand (2 reads)
            // leaving 4; the tail (no more reloads) tapers wait(4)->wait(2)->wait(0).
            // The 16-iter loop runs 2 flat mfmas/iter (j=2*iter, 2*iter+1) so the
            // online-softmax rescale stays per-iter (scales next iter's 8 oaccu).
            //
            // Rescale schedule (kDoRescale only): prologue pk_mul_pair scales
            // iter 0's 8 oaccu vgprs; in-loop iter i scales iter (i+1)'s 8 vgprs
            // as 4 mul_pair hidden under the V-tile load latency.
            // EVEN path: PV of THIS tile at call end (reads its own curr-pong V).
            // Odd defers PV to the next call's start (above) / the epilogue.
            if constexpr(kPvAtEnd && (kSkipCompute == false))
            {
                hk_mla_v40_pv_stage<kIsFirstIter, T>(kv_manager, p_lds_kv_curr, rescale,
                                                     do_rescale);
            }

            // ---- Epilogue ----
            //
            // Rescale oaccu by 1/row_sum_e (single mul_vgpr over full 128-vgpr
            // tile), then write 16-row x kVoHeadDim tile to vmem.
            //   partial_qo_loc < 0 -> final_output via OManager16bitsV4Gen1Swizzle (bf16).
            //   partial_qo_loc >= 0 -> split_output via OManager32bitsV4Gen1Swizzle (fp32)
            //                          + per-warp LSE row (lanes 0..15).
            // O LDS bounce overlays p_lds_kv_next (the next pong is dead on
            // the global last iter -- the swap is a no-op and the next
            // work_idx's KV prologue writes to p_lds_kv_curr).
            if constexpr(kDoEpilogue)
            {
                // ODD path drain: on the global-last call that ran a QK
                // (!kSkipCompute), THIS tile's PV is still pending (the rotation
                // defers PV by one). Do it now, reading V from p_lds_kv_curr
                // (this tile; no swap on the global last). The kSkipCompute
                // global-last case already drained the warp's last real tile via
                // the deferred PV at call start (kHasPv), so it needs nothing
                // here. Even path: PV already ran at call end.
                if constexpr((!kPvAtEnd) && (kSkipCompute == false))
                {
                    hk_mla_v40_pv_stage<false, T>(kv_manager, p_lds_kv_curr, rescale, do_rescale);
                }

                // ---- Attention-sink fold ----
                // Apply on OutputFinal (single-split == global) OR on the
                // LAST split of this batch element. By inflating exactly
                // one split's row_sum_e (and thus its lse), the reducer's
                // sum_k exp(lse_k - global_lse) * out_k formula naturally
                // routes exp(sink) into the global denominator exactly once
                // while contributing 0 to the V numerator.
                //
                // attn_sink is a per-lane VGPR loaded once at kernel entry
                // (-inf if p_attn_sink is null, so exp(...)=0 -> no-op).
                if(kEpilogueType == PvGemmEpilogueType::OutputFinal || is_last_split)
                {
                    // row_max is in log2 units (Q pre-scaled), attn_sink is a raw
                    // logit, so convert the sink to log2 units before the diff.
                    const float sink_term =
                        __builtin_amdgcn_exp2f(attn_sink * log2e - row_max);
                    row_sum_e += sink_term;
                }

                const comp_t reci_row_sum_e = 1.0f / row_sum_e;
                hk::mul_vgpr(oaccu, oaccu, reci_row_sum_e);

                const uintptr_t p_lds_o             = p_lds_kv_next;
                constexpr uint32_t num_pv_pair_iter = T::kVoHeadDim / (2u * T::kBlockN); // 8
                if constexpr(kEpilogueType == PvGemmEpilogueType::OutputFinal)
                {
                    opus::static_for<num_pv_pair_iter>([&](auto i) {
                        constexpr uint32_t iter       = i.value;
                        constexpr uint32_t kOaccuBase = k_o_begin + iter * 16u;
                        constexpr uint32_t kColOff    = iter * (2u * T::kBlockN);
                        o_manager.template output_to_vram_pair<kOaccuBase, kColOff, true>(
                            params.p_final_output, warp_idx, qo_start, qo_end, p_lds_o, num_qheads);
                        // Block LLVM from fusing adjacent OMgr calls' ds_reads
                        // (caps in-flight depth, keeps OMgr targets at v[58:69]).
                        __builtin_amdgcn_sched_barrier(0);
                    });
                }
                else
                {
                    opus::static_for<num_pv_pair_iter>([&](auto i) {
                        constexpr uint32_t iter       = i.value;
                        constexpr uint32_t kOaccuBase = k_o_begin + iter * 16u;
                        constexpr uint32_t kColOff    = iter * (2u * T::kBlockN);
                        split_o_manager.template output_to_vram_pair<kOaccuBase, kColOff, false>(
                            params.p_split_output,
                            warp_idx,
                            static_cast<uint32_t>(partial_qo_loc),
                            0,
                            p_lds_o,
                            num_qheads);
                        __builtin_amdgcn_sched_barrier(0);
                    });

                    // LSE: row_max + ln(row_sum_e). Lanes 0..15 own the M-rows
                    // after warp_reduce; lanes 16..63 hold redundant copies.
                    constexpr uint32_t kMfmaResultRows = 16;
                    if(lane_idx < kMfmaResultRows)
                    {
                        constexpr comp_t inv_log2e = 1.0f / log2e;
                        const uint32_t row_idx = lane_idx + warp_idx * kMfmaResultRows +
                                                 static_cast<uint32_t>(partial_qo_loc) * num_qheads;
                        // row_max is now in log2 units (Q pre-scaled).
                        // __builtin_amdgcn_logf == v_log_f32 == LOG2 in HW.
                        // lse_nat = row_max_nat + ln(sum)
                        //         = row_max_log2*inv_log2e + log2(sum)*inv_log2e
                        //         = (row_max + log2(sum)) * inv_log2e.
                        const comp_t lse =
                            (row_max + __builtin_amdgcn_logf(row_sum_e)) * inv_log2e;
                        params.p_split_lse[row_idx] = lse;
                    }
                }
            }

            // ---- Swap pongs ----
            // No-op on the global last iter (the swap-target is not consumed).
            if constexpr(kIsGlobalLast == false)
            {
                std::swap(p_lds_kv_curr, p_lds_kv_next);
            }
        };

        // ---- Per-warp dispatch ladder ----
        //
        // All warps execute the same number of global tiles. On tiles past
        // this warp's effective end (kv_end_eff), the warp dispatches mla_main
        // with kSkipCompute=true: still participates in barriers + cooperative
        // KV cvt+store but skips QK/softmax/PV. Epilogue fires only on the
        // global last tile and is synchronized across all working warps.
        //
        // Per-warp causal_offset < kBlockN (qseqlen <= 8, kBlockN = 32) means
        // num_iters_eff in {0, num_iters - 1, num_iters}: at most 1 trailing
        // skip iter. Same ladder shape as V32 m16x8.
#if !defined(MLA_SLIM_DISPATCH)
        if(kv_len_eff <= 0)
        {
            // Warp fully idle. num_iters == 1. One skip iter on the global
            // last tile, no epilogue (no oaccu state).
            mla_main.template operator()<false, true, PvGemmEpilogueType::None, false>(kv_start,
                                                                                       kv_end);
        }
        else if(kv_len_eff < T::kBlockN)
        {
            // Warp has exactly 1 partial real tile.
            if(kv_len < T::kBlockN)
            {
                // num_iters == 1: single real iter, also the epilogue iter.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start, kv_end);
                }
            }
            else
            {
                // num_iters == 2: real (partial) iter on tile 0, then
                // skip+epilogue on tile 1.
                mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(
                    kv_start, kv_start + T::kBlockN);
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start + T::kBlockN, kv_end);
                }
            }
        }
        else if(kv_len_eff == T::kBlockN)
        {
            // Warp has exactly 1 exact (full) real tile.
            if(kv_len == T::kBlockN)
            {
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start, kv_end);
                }
            }
            else
            {
                // num_iters == 2: exact real iter on tile 0, then skip+epilogue
                // on tile 1. kCheckBoundaryNext iff global last tile is partial.
                const bool boundary_next = (kv_len % T::kBlockN) != 0;
                if(boundary_next)
                {
                    mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(
                        kv_start, kv_start + T::kBlockN);
                }
                else
                {
                    mla_main.template operator()<true, false, PvGemmEpilogueType::None, false>(
                        kv_start, kv_start + T::kBlockN);
                }
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start + T::kBlockN, kv_end);
                }
            }
        }
        else // kv_len_eff > kBlockN: warp has >= 2 real tiles
        {
            const int32_t kv_1st_end = kv_start + T::kBlockN;

            // First real tile (kIsFirstIter=true). Next-tile boundary check
            // iff the tile being prefetched (tile 1) is the global last AND
            // partial.
            if((kv_1st_end + T::kBlockN - 1) < kv_end)
            {
                mla_main.template operator()<true, false, PvGemmEpilogueType::None, false>(
                    kv_start, kv_1st_end);
            }
            else
            {
                mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(
                    kv_start, kv_1st_end);
            }

            int32_t kv_idx = kv_1st_end;
            // Middle real tiles. Split the range so the inner loop only
            // contains iters whose NEXT tile is fully in bounds
            // (kCheckBoundaryNext=false, cheap). Any final middle iter
            // whose next tile may straddle the global end is handled
            // outside the loop with kCheckBoundaryNext=true. This avoids
            // a per-iter branch inside the hot middle loop (~2-3% perf
            // gain measured via thread trace).
            while((kv_idx + T::kBlockN) < kv_end_eff && (kv_idx + 2 * T::kBlockN) <= kv_end)
            {
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, false>(
                    kv_idx, kv_idx + T::kBlockN);
                kv_idx += T::kBlockN;
            }
            // Trailing middle iter (if any): its next tile is the global
            // last (possibly partial) -> boundary-checked prefetch.
            if((kv_idx + T::kBlockN) < kv_end_eff)
            {
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, true>(
                    kv_idx, kv_idx + T::kBlockN);
                kv_idx += T::kBlockN;
            }

            // Warp's last real tile starts at kv_idx. It may or may not
            // coincide with the global last tile.
            const bool tile_is_global_last = ((kv_idx + T::kBlockN) >= kv_end);

            if(tile_is_global_last)
            {
                // Warp's last real == global last -> real iter with epilogue.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_idx, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_idx, kv_end);
                }
            }
            else
            {
                // Warp's last real is NOT the global last; one trailing skip
                // iter does the epilogue. Real iter prefetches K for the
                // global last tile.
                const bool boundary_next = (kv_len % T::kBlockN) != 0;
                if(boundary_next)
                {
                    mla_main.template operator()<false, false, PvGemmEpilogueType::None, true>(
                        kv_idx, kv_idx + T::kBlockN);
                }
                else
                {
                    mla_main.template operator()<false, false, PvGemmEpilogueType::None, false>(
                        kv_idx, kv_idx + T::kBlockN);
                }
                // Skip + epilogue on the global last tile.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_idx + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_idx + T::kBlockN, kv_end);
                }
            }
        }
#else  // MLA_SLIM_DISPATCH
       // Slim dispatch: always use kCheckBoundaryNext=true. This drops the
       // kv_len%kBlockN==0 / kv_len_eff%kBlockN==0 fast-path
       // instantiations (rare in practice with random kv seqlens), halving
       // the number of template instantiations of mla_main. Cost: 1 cmp +
       // 1 cmov per K-iter for in_bounds check inside prefetch_kv_tile.
        if(kv_len_eff <= 0)
        {
            // Warp fully idle. Single skip iter, no epilogue.
            mla_main.template operator()<false, true, PvGemmEpilogueType::None, false>(kv_start,
                                                                                       kv_end);
        }
        else if(kv_len_eff <= T::kBlockN)
        {
            // Warp has exactly 1 real tile (full or partial).
            const bool tile_is_global_last = (kv_start + T::kBlockN) >= kv_end;
            if(tile_is_global_last)
            {
                // Real iter is also the epilogue iter; no next tile.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<true, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start, kv_end);
                }
            }
            else
            {
                // Real iter prefetches the global last tile (boundary-checked).
                mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(
                    kv_start, kv_start + T::kBlockN);
                // Trailing skip + epilogue.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_start + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_start + T::kBlockN, kv_end);
                }
            }
        }
        else // kv_len_eff > kBlockN: >= 2 real tiles
        {
            const int32_t kv_1st_end = kv_start + T::kBlockN;

            // First real iter; next prefetch boundary-checked.
            mla_main.template operator()<true, false, PvGemmEpilogueType::None, true>(kv_start,
                                                                                      kv_1st_end);

            int32_t kv_idx = kv_1st_end;
            // Middle real tiles. Split the range so the inner loop only
            // contains iters whose NEXT tile is fully in bounds
            // (kCheckBoundaryNext=false, cheap). Any final middle iter
            // whose next tile may straddle the global end is handled
            // outside the loop with kCheckBoundaryNext=true.
            while((kv_idx + T::kBlockN) < kv_end_eff && (kv_idx + 2 * T::kBlockN) <= kv_end)
            {
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, false>(
                    kv_idx, kv_idx + T::kBlockN);
                kv_idx += T::kBlockN;
            }
            // Trailing middle iter (if any): its next tile is the global
            // last (possibly partial) -> boundary-checked prefetch.
            if((kv_idx + T::kBlockN) < kv_end_eff)
            {
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, true>(
                    kv_idx, kv_idx + T::kBlockN);
                kv_idx += T::kBlockN;
            }

            // Warp's last real tile starts at kv_idx.
            const bool tile_is_global_last = ((kv_idx + T::kBlockN) >= kv_end);
            if(tile_is_global_last)
            {
                // Warp's last real == global last -> real iter with epilogue.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::OutputFinal, false>(
                            kv_idx, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, false, PvGemmEpilogueType::OutputSplit, false>(
                            kv_idx, kv_end);
                }
            }
            else
            {
                // Last real iter prefetches the global last tile (boundary-checked).
                mla_main.template operator()<false, false, PvGemmEpilogueType::None, true>(
                    kv_idx, kv_idx + T::kBlockN);
                // Skip + epilogue on the global last tile.
                if(partial_qo_loc < 0)
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputFinal, false>(
                            kv_idx + T::kBlockN, kv_end);
                }
                else
                {
                    mla_main
                        .template operator()<false, true, PvGemmEpilogueType::OutputSplit, false>(
                            kv_idx + T::kBlockN, kv_end);
                }
            }
        }
#endif // MLA_SLIM_DISPATCH
    }
}

template <typename T>
__global__ __launch_bounds__(T::kNumThreads, T::kOccupancy) __attribute__((amdgpu_num_vgpr(44)))
void kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(HkMlaV40DecodeFwdParams<T>
                                                                                   params)
{
    const uint32_t warp_idx = __builtin_amdgcn_readfirstlane(threadIdx.x / opus::get_warp_size());

    // Diverge on warp type ONCE at kernel entry so each type compiles as its own
    // body (compile-time kWarpType, no runtime warp-idx branches inside). Split
    // by HALF (warps 0-3 = Lo / PV-at-end, warps 4-7 = Hi / PV-deferred) so each
    // SIMD's (w, w+4) pair gets one of each path and the PVs stagger. RoPE
    // owners are 5,7 (both Hi).
    if(warp_idx < 4u)
    {
        mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1_impl<WarpType::LoNoPEWarp>(params,
                                                                                      warp_idx);
    }
    else if(KvManager8to16bitsV1<T>::wave_is_rope_owner(warp_idx))
    {
        mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1_impl<WarpType::HiRoPEWarp>(params,
                                                                                      warp_idx);
    }
    else
    {
        mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1_impl<WarpType::HiNoPEWarp>(params,
                                                                                      warp_idx);
    }
}

#else
template <typename T>
__global__ __launch_bounds__(
    T::kNumThreads,
    T::kOccupancy) void kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(HkMlaV40DecodeFwdParams<T>
                                                                                   params)
{
    (void)params;
    assert(false);
}
#endif

template <typename Traits>
void mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(aiter_tensor_t& query,
                                                         aiter_tensor_t& query_rope,
                                                         aiter_tensor_t& kv_buffer,
                                                         aiter_tensor_t& kv_buffer_rope,
                                                         const aiter_tensor_t& qo_indptr,
                                                         const aiter_tensor_t& kv_page_indices,
                                                         const aiter_tensor_t& kv_last_page_lens,
                                                         const aiter_tensor_t& work_indptr,
                                                         const aiter_tensor_t& work_info_set,
                                                         const int max_seqlen_q,
                                                         const float softmax_scale,
                                                         aiter_tensor_t& split_output,
                                                         aiter_tensor_t& split_lse,
                                                         aiter_tensor_t& final_output,
                                                         const float* p_attn_sink)
{
    // Shape / dtype / rank checks live ONCE in the outer dispatcher
    // (hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1) so we don't
    // pay for them per page_size template instantiation.
    const int32_t num_qheads      = query.size(1);
    const int32_t log2_num_qheads = __builtin_ctz(num_qheads);

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const hipStream_t stream = aiter::getCurrentHIPStream();

    HkMlaV40DecodeFwdParams<Traits> params = {
        reinterpret_cast<typename Traits::q_nope_t const*>(query.data_ptr()),
        reinterpret_cast<typename Traits::q_rope_t const*>(query_rope.data_ptr()),
        reinterpret_cast<typename Traits::kv_nope_t const*>(kv_buffer.data_ptr()),
        reinterpret_cast<typename Traits::kv_rope_t const*>(kv_buffer_rope.data_ptr()),
        // kv_indices
        reinterpret_cast<int32_t*>(kv_page_indices.data_ptr()),
        // kv_last_page_lens (only read by kernel when kPageSize > 1)
        reinterpret_cast<int32_t*>(kv_last_page_lens.data_ptr()),
        // metadata
        reinterpret_cast<int32_t*>(work_indptr.data_ptr()),
        reinterpret_cast<int32_t*>(work_info_set.data_ptr()),
        // optional per-head attention sink ([num_qheads] fp32, or nullptr)
        p_attn_sink,
        // outputs
        reinterpret_cast<typename Traits::out_t*>(final_output.data_ptr()),
        reinterpret_cast<float*>(split_output.data_ptr()),
        reinterpret_cast<float*>(split_lse.data_ptr()),
        // parameters
        softmax_scale,
        log2_num_qheads};

    const dim3 grid        = dim3(dev_prop.multiProcessorCount);
    const int32_t lds_size = dev_prop.maxSharedMemoryPerMultiProcessor / Traits::kOccupancy;

    kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1<Traits>
        <<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
}

void hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(aiter_tensor_t& query,
                                                            aiter_tensor_t& query_rope,
                                                            aiter_tensor_t& kv_buffer,
                                                            aiter_tensor_t& kv_buffer_rope,
                                                            const aiter_tensor_t& qo_indptr,
                                                            const aiter_tensor_t& kv_page_indices,
                                                            const aiter_tensor_t& kv_last_page_lens,
                                                            const aiter_tensor_t& work_indptr,
                                                            const aiter_tensor_t& work_info_set,
                                                            const int max_seqlen_q,
                                                            const float softmax_scale,
                                                            aiter_tensor_t& split_output,
                                                            aiter_tensor_t& split_lse,
                                                            aiter_tensor_t& final_output,
                                                            std::optional<aiter_tensor_t> attn_sink)
{
    HipDeviceGuard device_guard(final_output.device_id);

    const bool q_nope_is_fp8   = (query.dtype() == AITER_DTYPE_fp8);
    const bool kv_nope_is_fp8  = (kv_buffer.dtype() == AITER_DTYPE_fp8);
    const bool q_rope_is_bf16  = (query_rope.dtype() == AITER_DTYPE_bf16);
    const bool kv_rope_is_bf16 = (kv_buffer_rope.dtype() == AITER_DTYPE_bf16);

    AITER_CHECK(q_nope_is_fp8 && kv_nope_is_fp8,
                "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1 requires FP8 NOPE; got q=",
                AiterDtype_to_str(query.dtype()),
                ", kv=",
                AiterDtype_to_str(kv_buffer.dtype()));
    AITER_CHECK(
        q_rope_is_bf16 && kv_rope_is_bf16,
        "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1 requires BF16 ROPE; got q_rope=",
        AiterDtype_to_str(query_rope.dtype()),
        ", kv_rope=",
        AiterDtype_to_str(kv_buffer_rope.dtype()));

    // ---- Shape / rank checks ----
    // The kernel takes raw device pointers (no HK gl_* shape carrier), so
    // every shape MUST be validated here against the V4 layout constants:
    // any mismatch silently OOBs the kernel. Checks live ONCE in the outer
    // dispatcher (page_size-independent constants only) so the rank/size
    // logic doesn't bloat per page-size instantiation.
    //
    // Pull constants from a dummy traits instantiation so the values stay in
    // sync with HkMlaV40DecodeFwdTraits without duplication. kPageSize_=1 is
    // arbitrary -- only page_size-independent constants are used below.
    using DummyTraits = HkMlaV40DecodeFwdTraits<hk::fp8e4m3,
                                                hk::bf16,
                                                hk::fp8e4m3,
                                                hk::bf16,
                                                hk::bf16,
                                                /*kBlockN_=*/32,
                                                /*kNumWarps_=*/8,
                                                /*kOccupancy_=*/1,
                                                /*kBlockM_=*/128,
                                                /*kPageSize_=*/1>;

    const int64_t num_qheads = query.size(1);
    AITER_CHECK((num_qheads & (num_qheads - 1)) == 0 && num_qheads >= 16 && num_qheads <= 128,
                "num_qheads must be a power of 2 in [16, 128], got ",
                num_qheads);
    AITER_CHECK(num_qheads * max_seqlen_q == DummyTraits::kBlockM,
                "num_qheads * max_seqlen_q must equal ",
                DummyTraits::kBlockM,
                ", got ",
                num_qheads,
                " * ",
                max_seqlen_q,
                " = ",
                num_qheads * max_seqlen_q);

    AITER_CHECK(query.dim() == 3,
                "query must be 3-D [total_q, num_qheads, kQkPackedNopeQElems], got rank ",
                query.dim());
    AITER_CHECK(query.size(2) == DummyTraits::kQkPackedNopeQElems,
                "query.size(2) must equal kQkPackedNopeQElems=",
                DummyTraits::kQkPackedNopeQElems,
                ", got ",
                query.size(2));

    AITER_CHECK(query_rope.dim() == 3,
                "query_rope must be 3-D [total_q, num_qheads, kQkRopeHeadDim], got rank ",
                query_rope.dim());
    AITER_CHECK(query_rope.size(0) == query.size(0) && query_rope.size(1) == num_qheads,
                "query_rope dims 0,1 must match query: query=[",
                query.size(0),
                ",",
                query.size(1),
                "] vs query_rope=[",
                query_rope.size(0),
                ",",
                query_rope.size(1),
                "]");
    AITER_CHECK(query_rope.size(2) == DummyTraits::kQkRopeHeadDim,
                "query_rope.size(2) must equal kQkRopeHeadDim=",
                DummyTraits::kQkRopeHeadDim,
                ", got ",
                query_rope.size(2));

    const int32_t page_size = kv_buffer.size(1);

    AITER_CHECK(kv_buffer.dim() == 4,
                "kv_buffer must be 4-D [num_page, page_size, kKvNumHead, kQkPackedNopeKvElems], "
                "got rank ",
                kv_buffer.dim());
    AITER_CHECK(kv_buffer.size(2) == DummyTraits::kKvNumHead,
                "kv_buffer.size(2) must equal kKvNumHead=",
                DummyTraits::kKvNumHead,
                ", got ",
                kv_buffer.size(2));
    AITER_CHECK(kv_buffer.size(3) == DummyTraits::kQkPackedNopeKvElems,
                "kv_buffer.size(3) must equal kQkPackedNopeKvElems=",
                DummyTraits::kQkPackedNopeKvElems,
                ", got ",
                kv_buffer.size(3));

    AITER_CHECK(
        kv_buffer_rope.dim() == 4, "kv_buffer_rope must be 4-D, got rank ", kv_buffer_rope.dim());
    AITER_CHECK(
        kv_buffer_rope.size(0) == kv_buffer.size(0) && kv_buffer_rope.size(1) == page_size &&
            kv_buffer_rope.size(2) == DummyTraits::kKvNumHead,
        "kv_buffer_rope dims 0..2 must match kv_buffer's [num_page, page_size, kKvNumHead]=[",
        kv_buffer.size(0),
        ",",
        page_size,
        ",",
        DummyTraits::kKvNumHead,
        "], got [",
        kv_buffer_rope.size(0),
        ",",
        kv_buffer_rope.size(1),
        ",",
        kv_buffer_rope.size(2),
        "]");
    AITER_CHECK(kv_buffer_rope.size(3) == DummyTraits::kQkRopeHeadDim,
                "kv_buffer_rope.size(3) must equal kQkRopeHeadDim=",
                DummyTraits::kQkRopeHeadDim,
                ", got ",
                kv_buffer_rope.size(3));

    AITER_CHECK(final_output.dim() == 3,
                "final_output must be 3-D [total_q, num_qheads, kVoHeadDim], got rank ",
                final_output.dim());
    AITER_CHECK(final_output.size(0) == query.size(0) && final_output.size(1) == num_qheads &&
                    final_output.size(2) == DummyTraits::kVoHeadDim,
                "final_output shape must be [",
                query.size(0),
                ",",
                num_qheads,
                ",",
                DummyTraits::kVoHeadDim,
                "], got [",
                final_output.size(0),
                ",",
                final_output.size(1),
                ",",
                final_output.size(2),
                "]");

    AITER_CHECK(split_output.dim() >= 2 &&
                    split_output.size(split_output.dim() - 1) == DummyTraits::kVoHeadDim,
                "split_output trailing dim must equal kVoHeadDim=",
                DummyTraits::kVoHeadDim,
                ", got ",
                split_output.size(split_output.dim() - 1));
    AITER_CHECK(split_lse.dim() >= 1, "split_lse must have rank >= 1");

    AITER_CHECK(work_indptr.dim() == 1, "work_indptr must be 1-D, got rank ", work_indptr.dim());
    AITER_CHECK(kv_page_indices.dim() == 1,
                "kv_page_indices must be 1-D, got rank ",
                kv_page_indices.dim());

    // Optional attention sink: [num_qheads] fp32. Disabled when absent.
    const float* p_attn_sink = nullptr;
    if(attn_sink.has_value())
    {
        const aiter_tensor_t& s = attn_sink.value();
        AITER_CHECK(s.dtype() == AITER_DTYPE_fp32,
                    "attn_sink must be fp32, got ",
                    AiterDtype_to_str(s.dtype()));
        AITER_CHECK(s.dim() == 1, "attn_sink must be 1-D, got rank ", s.dim());
        AITER_CHECK(s.size(0) == num_qheads,
                    "attn_sink.size(0) must equal num_qheads=",
                    num_qheads,
                    ", got ",
                    s.size(0));
        p_attn_sink = reinterpret_cast<const float*>(s.data_ptr());
    }

#define DISPATCH_PAGE_SIZE(PageSize)                                                   \
    case PageSize: {                                                                   \
        using Traits = HkMlaV40DecodeFwdTraits<hk::fp8e4m3,                            \
                                               hk::bf16,                               \
                                               hk::fp8e4m3,                            \
                                               hk::bf16,                               \
                                               hk::bf16,                               \
                                               /*kBlockN_=*/32,                        \
                                               /*kNumWarps_=*/8,                       \
                                               /*kOccupancy_=*/1,                      \
                                               /*kBlockM_=*/128,                       \
                                               /*kPageSize_=*/PageSize>;               \
        mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1<Traits>(query,             \
                                                                    query_rope,        \
                                                                    kv_buffer,         \
                                                                    kv_buffer_rope,    \
                                                                    qo_indptr,         \
                                                                    kv_page_indices,   \
                                                                    kv_last_page_lens, \
                                                                    work_indptr,       \
                                                                    work_info_set,     \
                                                                    max_seqlen_q,      \
                                                                    softmax_scale,     \
                                                                    split_output,      \
                                                                    split_lse,         \
                                                                    final_output,      \
                                                                    p_attn_sink);      \
        break;                                                                         \
    }

    // Only page_size in {1, 64} are instantiated -- same pattern as v32.
    switch(page_size)
    {
        DISPATCH_PAGE_SIZE(1)
        DISPATCH_PAGE_SIZE(64)
    default:
        AITER_CHECK(
            false,
            "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1: unsupported page_size ",
            page_size,
            " (supported: 1, 64).");
    }

#undef DISPATCH_PAGE_SIZE
}
