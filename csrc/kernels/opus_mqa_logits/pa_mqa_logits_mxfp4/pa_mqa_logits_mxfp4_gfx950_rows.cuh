// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 wave64 rows kernel (V4.1 full layers, 32 heads): ROWS_PER_WAVE query rows of one sequence
// a wave, MFMA 32x32x64 -- a row's 32 heads are one m-tile, a 64-key tile two n-tiles of 32, D two
// k-tiles of 64. Device pass on gfx950 only (pa_mqa_logits_mxfp4_gfx950.cu).
//
// The kernel is VALU-bound: a relu and an fma a (head, key, row), against 1/16 of a cycle of MFMA.
// The MFMA is 32x32x64 so it hides under that VALU: every MFMA issue costs the wave a few cycles
// of VALU issue, and the long shape needs the fewest issues for the FLOPs.
//
// A tile's work is R x 2 items (row r, n-tile j) of two MFMAs (k-tiles 0 and 1 into one
// accumulator), software-pipelined by one item: item k's MFMAs around item k - 1's relu and
// weighted head sum, so two items' accumulators are live and a row costs no registers but its
// Q (8), q_scale (1) and weights (1) -- up to eight rows a wave.
//
// Lane L = g * 32 + i (i = L % 32, g = L / 32); A and B hold D block kt * 2 + g:
//   A (Q, row r, k-tile kt)  head i: 16 B at row * 2048 + i * 64 + (kt * 2 + g) * 16
//   B (KV, n-tile j, kt)     key 32 j + i: group + (kt * 2 + g) * 1024 + (32 j + i) * 16
//   C (row r, n-tile j)      heads rept * 8 + g * 4 + pack (16 of them), key 32 j + i
// q_scale [T][g][m][kt]: lane L's short is element L, byte kt (op_sel kt). kv_scale
// [groups][g][m][kt * 2 + tile]: lane L's dword is element L, byte kt * 2 + j (op_sel).
// A row's two n-tiles' head sums meet across the lane groups in one permlane32 swap, which leaves
// tile key L in lane L: one 64-lane store a row a tile.
#pragma once
#include "pa_mqa_logits_mxfp4_gfx950_common.cuh"

namespace opus_logits::gfx950 {

template<class T, int SCHED>
__global__ __launch_bounds__(T::BLOCK_SIZE, T::MIN_WAVES)
void rows_kernel(mqa_logits_mxfp4_kargs k) {
    using D_DATA   = opus::fp4_t;
    using D_BYTE   = uint8_t;
    using D_WEIGHT = typename T::D_WEIGHT;
    using D_OUT    = typename T::D_OUT;
    using Mma      = opus::mfma<D_DATA, D_DATA, float, T::MMA_M, T::MMA_N, T::MMA_K, T::WAVE_SIZE>;
    using va_t     = typename Mma::vtype_a;
    using vb_t     = typename Mma::vtype_b;
    using vc_t     = typename Mma::vtype_c;

    constexpr int R     = T::ROWS_PER_WAVE;
    constexpr int NT    = T::N_TILES;   // 2 n-tiles of 32 keys a tile
    constexpr int KT    = T::K_TILES;   // 2 k-tiles of 64
    constexpr int EC    = T::C_FRAG;    // 16 heads a lane an item
    constexpr int TK    = T::KV_TILE_SIZE;
    constexpr int BUF   = T::STAGES;    // KV tiles in registers: the one computed + in flight
    constexpr int SW    = T::SLOT_WAVES;
    constexpr int ITEMS = R * NT;       // a tile's (row, n-tile) items, row-major
    static_assert(Mma::elem_c == EC && ITEMS % 2 == 0, "two accumulator slots, alternating");
    Mma mma;

    // One scalar round trip, pinned ahead of the early exits.
    const void* p_q        = k.ptr_q;
    const void* p_q_scale  = k.ptr_q_scale;
    const void* p_kv       = k.ptr_kv;
    const void* p_kv_scale = k.ptr_kv_scale;
    const int*  p_bt       = k.ptr_block_tables;
    const void* p_weights  = k.ptr_weights;
    float*      p_out      = k.ptr_out;
    const int*  p_out_offs = k.ptr_out_offsets;
    int   stride_out   = k.out_stride;
    float weight_scale = k.weight_scale;
    int   bt_stride    = k.bt_stride;
    int   tpb_shift    = k.tpb_shift;
    mqa_sched_args sa(k);
    pin_sgpr(sa.num_rows);
    pin_sgpr(p_q); pin_sgpr(p_q_scale); pin_sgpr(p_kv); pin_sgpr(p_kv_scale);
    pin_sgpr(p_bt); pin_sgpr(p_weights); pin_sgpr(p_out); pin_sgpr(p_out_offs);
    pin_sgpr(stride_out); pin_sgpr(weight_scale); pin_sgpr(bt_stride); pin_sgpr(tpb_shift);

    const int tid  = opus::thread_id_x();
    const int lane = SW == 1 ? tid : tid % T::WAVE_SIZE;
    const int wave = SW == 1 ? 0 : __builtin_amdgcn_readfirstlane(tid / T::WAVE_SIZE);
    const int li   = lane % T::MMA_N;   // head (A, C) / key (B, C) within a tile
    const int lg   = lane / T::MMA_N;   // D block within a k-tile (A / B), head group (C)

    // Wave-uniform; past here no wave waits on another.
    mqa_work wk;
    if (!mqa_assign<R, TK, SCHED, false, SW>(sa, lane, wk, wave)) return;
    const int row_id      = wk.row_id;
    const int chunk_start = wk.chunk_start;
    const int tile_count  = wk.chunk_count;
    const int group_rows  = wk.group_rows;

    // ---- the rows: Q, scales, weights, an out window each ----
    // A slot past the group computes its first row's and stores nothing (an empty window).
    //
    // A row's 16 weights a lane are one register: lane 16 k + q holds slot q of lane group k / 2
    // (both 16-lane rows of a lane group hold its 16), read by the fma's DPP row_newbcast:q --
    // 16 registers a row fewer, at the cost of a DPP fma (as fast as a plain one under the MFMA).
    va_t     q_a[R][KT]{};   // zeroed: fp4 fills the low 16 B of the 256-bit operand
    int      q_sc[R];
    float    w_bc[R];
    D_OUT*   out_ptr[R];
    unsigned out_bytes[R];
    const int w_slot = lane % 16, w_group = lane / 32;
    const int w_head = (w_slot / 4) * 8 + w_group * 4 + w_slot % 4;   // C row of slot w_slot
    opus::static_for<R>([&](auto rc) {
        constexpr int r = rc.value;
        const bool ok  = r == 0 || r < group_rows;
        const int  row = row_id + (ok ? r : 0);
        const int  end = R == 1 ? wk.local_end : ok ? scalar_load(sa.row_ends, row) : 0;
        auto g_q = opus::make_gmem(reinterpret_cast<const D_BYTE*>(p_q) +
                                   (size_t)row * T::Q_ROW_BYTES);
        opus::static_for<KT>([&](auto kc) {
            constexpr int kt = kc.value;
            auto v = opus::load<T::KV_GRP_BYTES>(
                g_q, li * T::Q_HEAD_BYTES + (kt * 2 + lg) * T::KV_GRP_BYTES);
            opus::set_slice(q_a[r][kt], v, opus::number<0>{}, opus::number<T::KV_GRP_BYTES>{});
        });
        auto g_qs = opus::make_gmem(reinterpret_cast<const unsigned short*>(p_q_scale) +
                                    (size_t)row * T::QS_ROW_SHORTS);
        q_sc[r] = opus::load<1>(g_qs, lane)[0];
        auto g_w = opus::make_gmem(reinterpret_cast<const D_WEIGHT*>(p_weights) +
                                   (size_t)row * T::W_ROW_ELEMS);
        w_bc[r] = (float)opus::load<1>(g_w, w_head)[0] * weight_scale;
        out_ptr[r]   = row_out<T>(p_out, p_out_offs, stride_out, row);
        out_bytes[r] = (unsigned)(end > 0 ? end : 0) * (unsigned)sizeof(D_OUT);
    });

    // ---- the KV walk ----
    // Tile t: its entry's group, or (an entry a run of 2^tpb_shift groups) the run's t-th; the
    // table through the constant address space, an s_load. pid[] holds the raw entry, read BUF
    // tiles ahead, and its page is made where the KV is issued: made where it is read, the shift
    // waits on that s_load at once (SMEM returns out of order: every wait is lgkmcnt(0)).
    using bt_scalar_ptr = const int __attribute__((address_space(4)))*;
    auto bt_c = reinterpret_cast<bt_scalar_ptr>(
        reinterpret_cast<__UINTPTR_TYPE__>(p_bt + (size_t)wk.batch_id * bt_stride));
    const unsigned tpb_mask = (1u << tpb_shift) - 1;
    auto tile_at = [&](int t) {   // t clamped into the chunk: prefetches past it read its last
        return (unsigned)chunk_start + (unsigned)(t < tile_count ? t : tile_count - 1);
    };
    auto entry_of = [&](int t) { return bt_c[tile_at(t) >> tpb_shift]; };
    auto page_at = [&](int entry, int t) {
        return (int)(((unsigned)entry << tpb_shift) | (tile_at(t) & tpb_mask));
    };
    const int lane_kv = lg * T::STRIDE_KV_G + li * T::STRIDE_KV_M;   // block g, key i

    struct kv_tile {
        vb_t b[NT][KT];
        int  d;   // the lane's key-scale dword: byte kt * 2 + j
    };
    kv_tile kv[BUF];
    opus::static_for<BUF>([&](auto bc) {
        opus::static_for<NT>([&](auto jc) {
            opus::static_for<KT>([&](auto kc) { kv[bc.value].b[jc.value][kc.value] = vb_t{}; });
        });
    });
    // KV_AUX: the loads' cache policy (2: nontemporal, for walks no other CTA reads again). A V# a
    // page, its base 64-bit scalar arithmetic off the (wave-uniform) page: the offsets stay inside
    // the page, so no 32-bit offset bounds the kv_cache.
    constexpr auto KV_AUX = opus::number<T::KV_AUX>{};
    auto issue = [&](kv_tile& x, int page) {
        auto g_kv  = opus::make_gmem(reinterpret_cast<const D_BYTE*>(p_kv) +
                                     (size_t)(unsigned)page * T::KV_PAGE_BYTES);
        auto g_kvs = opus::make_gmem(reinterpret_cast<const int*>(p_kv_scale) +
                                     (size_t)(unsigned)page * (T::KVS_PAGE_BYTES / 4));
        opus::static_for<KT>([&](auto kc) {
            constexpr int kt = kc.value;
            opus::static_for<NT>([&](auto jc) {
                constexpr int j = jc.value;
                auto v = opus::load<T::KV_GRP_BYTES>(
                    g_kv, lane_kv + kt * 2 * T::STRIDE_KV_G + j * T::MMA_N * T::STRIDE_KV_M, 0,
                    KV_AUX);
                opus::set_slice(x.b[j][kt], v, opus::number<0>{},
                                opus::number<T::KV_GRP_BYTES>{});
            });
        });
        x.d = opus::load<1>(g_kvs, lane, 0, KV_AUX)[0];
    };

    // ---- an item: two MFMAs; its epilogue: relu x weight over the lane's 16 heads ----
    float part[R][NT];   // a lane's head sum, (row, n-tile)
    vc_t  acc[2];        // two items' accumulators, alternating
    // Heads [Q0, Q1) of item (r, j)'s sum, carried in part[r][j]: one chain of scalar fmas, not
    // v_pk_fma_f32 -- on gfx950 a packed FP32 op does not issue while the wave's MFMA is in
    // flight, a scalar one does. asm, or the SLP vectorizer packs them again; it reads only VALU
    // results, so it needs none of the MFMA -> VALU wait states the compiler inserts, and its
    // weight register is written once, ahead of the walk, so the DPP read needs none either. The
    // asm also fixes the order: the TU is built with -ffast-math, and the epilogue is inlined at
    // several places, each of which a reassociated chain could round differently -- and which one
    // computes a tile depends on where the schedule cut the walk.
    auto epilogue = [&](auto rc, auto jc, auto sc, auto q0c, auto q1c) {
        constexpr int r = decltype(rc)::value, j = decltype(jc)::value, s = decltype(sc)::value;
        constexpr int Q0 = decltype(q0c)::value, Q1 = decltype(q1c)::value;
        float& a = part[r][j];
        opus::static_for<Q1 - Q0>([&](auto qi) {
            constexpr int q = Q0 + qi.value;
            const float v  = relu(acc[s][q]);
            const float wq = w_bc[r];
            float& ac = a;   // asm operands name locals of this lambda only
            if constexpr (q == 0)
                asm("v_mul_f32_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf"
                    : "=v"(ac) : "v"(wq), "v"(v));
            else
                asm("v_fmac_f32_dpp %0, %1, %2 row_newbcast:%3 row_mask:0xf bank_mask:0xf"
                    : "+v"(ac) : "v"(wq), "v"(v), "i"(q));
        });
    };
    // Row r's tile: lanes < 32 add n-tile 0's two head halves (key L), lanes >= 32 n-tile 1's.
    auto finish = [&](auto rc, int t) {
        constexpr int r = decltype(rc)::value;
        opus::vector_t<opus::u32_t, 2> x = __builtin_amdgcn_permlane32_swap(
            std::bit_cast<opus::u32_t>(part[r][0]), std::bit_cast<opus::u32_t>(part[r][1]),
            false, false);
        const float logit = std::bit_cast<float>(x[0]) + std::bit_cast<float>(x[1]);
        auto g_out = opus::make_gmem(out_ptr[r], out_bytes[r]);
        g_out.template store<1>(logit, (chunk_start + t) * TK + lane);
    };

    // Item `it` of tile x, k-tile kt, into acc[it % 2].
    auto mfma_item = [&](const kv_tile& x, auto itc, auto ktc) {
        constexpr int it = decltype(itc)::value, kt = decltype(ktc)::value;
        constexpr int r = it / NT, j = it % NT, s = it % 2;
        acc[s] = mma(q_a[r][kt], x.b[j][kt], kt == 0 ? vc_t{} : acc[s], q_sc[r], x.d,
                     opus::number<kt>{}, opus::number<kt * NT + j>{});
    };
    // Item `it` of tile x's two MFMAs, with item `pit`'s epilogue (of tile tp) around the second.
    // Fenced in that order: left alone, the scheduler hoists the epilogue above the MFMAs to share
    // their registers, and each epilogue then waits on its own MFMAs.
    constexpr int SPLIT = 4;   // heads before the second MFMA: 8 VALU, the first MFMA's span
    auto step = [&](const kv_tile& x, auto itc, auto pitc, int tp) {
        constexpr int pit = decltype(pitc)::value;
        constexpr int pr = pit / NT, pj = pit % NT, ps = pit % 2;
        __builtin_amdgcn_sched_barrier(0);
        mfma_item(x, itc, opus::number<0>{});
        __builtin_amdgcn_sched_barrier(0);
        epilogue(opus::number<pr>{}, opus::number<pj>{}, opus::number<ps>{}, opus::number<0>{},
                 opus::number<SPLIT>{});
        __builtin_amdgcn_sched_barrier(0);
        mfma_item(x, itc, opus::number<1>{});
        __builtin_amdgcn_sched_barrier(0);
        epilogue(opus::number<pr>{}, opus::number<pj>{}, opus::number<ps>{},
                 opus::number<SPLIT>{}, opus::number<EC>{});
        if constexpr (pj == NT - 1) finish(opus::number<pr>{}, tp);
    };

    // Prologue: tiles 0 .. BUF - 2 in flight, tile 0's first item's MFMAs.
    int pid[BUF];
    opus::static_for<BUF>([&](auto bc) { pid[bc.value] = entry_of(bc.value); });
    opus::static_for<BUF - 1>([&](auto bc) {
        constexpr int u = bc.value;
        issue(kv[u], page_at(pid[u], u));
        pid[u] = entry_of(u + BUF);
    });
    mfma_item(kv[0], opus::number<0>{}, opus::number<0>{});
    mfma_item(kv[0], opus::number<0>{}, opus::number<1>{});

    // Tile t: issue tile t + BUF - 1 into the buffer tile t - 1 freed, items 1 .. ITEMS - 1, then
    // tile t + 1's first item (past the walk: its last tile again, computed and dropped) under
    // tile t's last epilogue. Unrolled by BUF so every buffer index is a compile-time constant.
    for (int tb = 0; tb < tile_count; tb += BUF) {
        opus::static_for<BUF>([&](auto bc) {
            constexpr int b = bc.value, u = (b + BUF - 1) % BUF;
            const int t = tb + b;
            if (t >= tile_count) return;   // wave-uniform
            issue(kv[u], page_at(pid[u], t + BUF - 1));
            pid[u] = entry_of(t + 2 * BUF - 1);
            opus::static_for<ITEMS - 1>([&](auto kc) {
                constexpr int it = kc.value + 1;
                step(kv[b], opus::number<it>{}, opus::number<it - 1>{}, t);
            });
            step(kv[(b + 1) % BUF], opus::number<0>{}, opus::number<ITEMS - 1>{}, t);
        });
    }
}

}  // namespace opus_logits::gfx950
