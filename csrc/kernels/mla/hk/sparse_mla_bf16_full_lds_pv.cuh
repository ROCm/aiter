// SPDX-License-Identifier: MIT
// Included inside namespace sparse_mla_bf16, after FullLdsRegs.
// PR PV four-slot schedule, compact register map and D256 per wave.
// ColBase chooses the sb8-preserving physical output half.
template <uint32_t ColBase, bool kIsFirstIter, bool kDoRescale, typename T>
__device__ __forceinline__ void
full_lds_pv_gemm(Bf16KvManager<T>& kv_manager, const uintptr_t p_lds_v, const float rescale)
{
    using R                       = FullLdsRegs;
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
    constexpr uint32_t kDIters     = 256u / (2u * T::kTileM); // 8 per D256 wave
    constexpr uint32_t num_pv_iter = R::kNumKvSub * kDIters;  // 16 or 32
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
        kv_manager.template load_transposed_v_to_gpr<kRowBase + 0u, ColBase + jj * 16u, base + 0>(
            p_lds_v);
        kv_manager.template load_transposed_v_to_gpr<kRowBase + 16u, ColBase + jj * 16u, base + 2>(
            p_lds_v);
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
