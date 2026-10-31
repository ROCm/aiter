// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// The n_heads = 64 (DeepSeek-V4) kernel on gfx1250: 64-row pages, one table entry a 64-key tile.
// Device pass on gfx1250 only (pa_mqa_logits_mxfp4_gfx1250.cu). Where a CTA's work comes from:
// template SCHED, pa_mqa_logits_mxfp4_gfx1250_common.cuh.
//
//   * A tile's accumulators are pipelined by n-tile -- n-tile m's two WMMAs (one an m-tile), then
//     n-tile m-1's two epilogues -- so four are live, not two whole tiles': under 256 VGPRs.
//   * The pair finish: n-tiles 2u and 2u+1 share one permlane16_swap and one add, and all 32
//     lanes store a logit.
//   * Stores at the tile's end; the relu x weight sum as two pk_fma chains an accumulator; in
//     one-row CTAs the row's Q ahead of the first tiles' TDM, in four-row CTAs behind it; the
//     block-table word one tile ahead; the TDM descriptors built once, a tile patching its
//     address and stage.
//   * Wave priority by progress in one-round launches: a Table schedule's (the record's
//     chunk_max, see the builder) and Free's, never Prefill's (there the age order retires CTAs
//     early for the next round).
#pragma once
#include "pa_mqa_logits_mxfp4_gfx1250_common.cuh"

namespace opus_logits::gfx1250 {

template<class T, int SCHED>
__global__ __launch_bounds__(T::BLOCK_SIZE, 1)
void h64_kernel(mqa_logits_mxfp4_kargs k) {
    using D_BYTE   = opus::u8_t;
    using D_SCALE  = typename T::D_SCALE;
    using D_WEIGHT = typename T::D_WEIGHT;
    using Mma =
        opus::wmma<opus::fp4_t, opus::fp4_t, float, T::MMA_M, T::MMA_N, T::MMA_K, T::WAVE_SIZE>;
    Mma mma;
    constexpr int MT     = T::M_TILES;            // 2
    constexpr int NT     = T::N_TILES;            // 4
    constexpr int EC     = T::C_FRAG;             // 16
    constexpr int WAVE   = T::WAVE_SIZE;
    constexpr int PACK_W = T::PACK_C;             // 8 bf16 = one 16 B load
    constexpr int DW     = (int)sizeof(D_SCALE);
    constexpr int GRP_DW = T::KV_GRP_BYTES / DW;
    constexpr int TK     = T::KV_TILE_SIZE;       // 64
    constexpr int STAGES = T::STAGES;
    static_assert(MT == 2 && NT == 4 && T::PAGES_PER_TILE == 1,
                  "H = 64 is two m-tiles, and a 64-key tile is one page");
    // The first TDM_ISSUE_WAVES waves each load one piece of the tile; the rest (6 waves: 2) only
    // compute. A wave that issued nothing passes its tensorcnt wait at once, and the CTA barrier
    // after the wait is what orders every piece before any wave's read.
    static_assert(T::SCALE_TDM || T::TDM_ISSUE_WAVES == T::NUM_WAVES,
                  "lane-loaded scales are per wave: every wave runs issue_tile");

    __shared__ __attribute__((aligned(16))) D_BYTE smem_buf[T::LDS_BYTES];

    const int tid     = opus::thread_id_x();
    const int wave    = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int lane    = tid % WAVE;
    const int lane_m0 = lane % T::GRPN_C;
    const int lane_g  = lane / T::GRPN_C;

    int   num_rows     = k.num_rows;
    int   stride_out   = k.out_stride;
    float weight_scale = k.weight_scale;
    int   bt_stride    = k.bt_stride;
    pin_sgpr(num_rows); pin_sgpr(stride_out); pin_sgpr(weight_scale); pin_sgpr(bt_stride);

    // The group's window is its union and alone sets the trip count (the barrier count); a row's
    // own window only masks its stores. A wave past a short group runs with an empty window, so it
    // stores nothing and the barrier count stays uniform.
    mqa_work wk;
    if (!mqa_assign<T::NUM_WAVES, TK, false, SCHED, true>(k, lane, wk)) return;   // CTA-uniform
    const bool slot_ok     = wave < wk.group_rows && wk.row_id + wave < num_rows;
    const int  row_id      = wk.row_id + (slot_ok ? wave : 0);
    int        batch_id    = wk.batch_id;
    const int  chunk_start = wk.chunk_start;                  // absolute tile index
    const int  tile_count  = wk.chunk_count;
    const int  chunk_max   = SCHED == SCHED_PREFILL ? 0 : wk.chunk_max;   // 0: no priority
    pin_sgpr(batch_id);

    // ---- the tile's page: block table -> TDM into the LDS ring ----
    // The table through the constant address space so page ids come from s_load (TDM descriptor
    // fields live in SGPRs). A plain load cannot be proven NOCLOBBER past the store to out.
    using bt_scalar_ptr = const int __attribute__((address_space(4)))*;
    auto bt_c = reinterpret_cast<bt_scalar_ptr>(reinterpret_cast<__UINTPTR_TYPE__>(
        k.ptr_block_tables + (size_t)batch_id * bt_stride + chunk_start));

    // CU scope, not opus's device default (which defeats the L2). Legal only because KV and
    // scales are read-only for the launch: nothing may write the KV cache during the launch.
    using fp4x2 = opus::array<opus::fp4_t, 2>;
    constexpr int TDM_CACHE = opus::tdm_traits::make_cache_policy(
        opus::tdm_traits::load_temporal_hint::regular, opus::tdm_traits::scope::cu);
    using Cache  = opus::tdm_traits::cache<TDM_CACHE>;
    using Pad    = opus::tdm_traits::padding<fp4x2, T::LDS_PAD_INTERVAL, T::LDS_PAD_AMOUNT>;
    using KvWin  = opus::tdm<fp4x2, opus::seq<T::TOKEN_BYTES, T::TDM_ROWS>, Pad, Cache>;
    using KvsWin = opus::tdm<opus::u8_t, opus::seq<T::SCALE_BYTES_PER_DWORD, T::TDM_ROWS>, Cache>;
    constexpr opus::u32_t TDM_EXTENT = 1u << 30;   // rows; the address moves, the extent clamps
    const int tdm_piece    = wave;                 // one piece of the 64-row page a wave
    const int kv_lds_base  = tdm_piece * T::LDS_PIECE_BYTES;
    const int kvs_lds_base = T::lds_scale_byte(tdm_piece * T::TDM_ROWS);
    auto kv_win = opus::make_tdm<KvWin>(
        (opus::u32_t)kv_lds_base, reinterpret_cast<const D_BYTE*>(k.ptr_kv),
        (opus::u32_t)T::TOKEN_BYTES, TDM_EXTENT, (opus::u64_t)T::TOKEN_BYTES, 0u, 0u);
    auto kvs_win = opus::make_tdm<KvsWin>(
        (opus::u32_t)kvs_lds_base, reinterpret_cast<const D_BYTE*>(k.ptr_kv_scale),
        (opus::u32_t)T::SCALE_BYTES_PER_DWORD, TDM_EXTENT,
        (opus::u64_t)T::SCALE_BYTES_PER_DWORD, 0u, 0u);
    // Each window's D# at origin 0, built once; a tile's issue patches the global address and
    // the LDS stage (rebuilding the D# costs ~25 dependent SALU an op).
    const typename KvWin::descriptor  kv_d0  = kv_win.make_descriptor();
    const typename KvsWin::descriptor kvs_d0 = kvs_win.make_descriptor();
    const opus::u64_t kv_g  = reinterpret_cast<opus::u64_t>(k.ptr_kv);
    const opus::u64_t kvs_g = reinterpret_cast<opus::u64_t>(k.ptr_kv_scale);
    // Tile t (page id `pid`) into ring slot `slot`; nothing is issued past the chunk (the wait
    // below drains instead), so a short chunk fetches no duplicate pages.
    int kvs_ring[STAGES][NT / 2];
    auto issue_tile = [&](int t, auto slot_c, int pid) {
        constexpr int slot = decltype(slot_c)::value;
        if (t >= tile_count) return;
        if (T::TDM_ISSUE_WAVES < T::NUM_WAVES && wave >= T::TDM_ISSUE_WAVES) return;   // uniform
        const opus::u32_t row0   = (opus::u32_t)pid * T::PAGE_SIZE;
        const opus::u32_t origin = row0 + tdm_piece * T::TDM_ROWS;
        const opus::u32_t stage  = (opus::u32_t)(slot * T::LDS_STAGE_BYTES);
        {
            auto d = kv_d0;
            d.set_lds_addr((opus::u32_t)kv_lds_base + stage);
            d.set_global_addr(kv_g + (opus::u64_t)origin * T::TOKEN_BYTES);
            KvWin::async_load(d);
        }
        if constexpr (T::SCALE_TDM) {
            auto d = kvs_d0;
            d.set_lds_addr((opus::u32_t)kvs_lds_base + stage);
            d.set_global_addr(kvs_g + (opus::u64_t)origin * T::SCALE_BYTES_PER_DWORD);
            KvsWin::async_load(d);
        } else {
            // lane L, half u: key 32u + L of the tile, i.e. n-tiles 2u (low lanes) and 2u+1 (high)
            opus::static_for<NT / 2>([&](auto uc) {
                constexpr int u = uc.value;
                kvs_ring[slot][u] = *reinterpret_cast<const int*>(
                    reinterpret_cast<const D_BYTE*>(k.ptr_kv_scale) +
                    ((size_t)(row0 + u * 2 * T::MMA_N + lane)) * T::SCALE_BYTES_PER_DWORD);
            });
        }
    };

    auto s_dw = opus::make_smem(reinterpret_cast<D_SCALE*>(smem_buf));
    const int lds_lane_dw = T::lds_byte(lane_m0, lane_g) / DW;
    auto read_kv = [&](typename Mma::vtype_b (&kv)[NT], int (&kvs)[NT / 2], auto slot_c) {
        constexpr int slot = decltype(slot_c)::value;
        const int stage_dw = slot * T::LDS_STAGE_BYTES / DW;
        opus::static_for<NT / 2>([&](auto uc) {
            constexpr int u = uc.value;
            if constexpr (T::SCALE_TDM)
                kvs[u] = opus::load<1>(
                    s_dw, stage_dw + T::lds_scale_byte(u * 2 * T::MMA_N) / DW + lane)[0];
            else
                kvs[u] = kvs_ring[slot][u];
        });
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt   = ntc.value;
            constexpr int base = nt * T::LDS_NTILE_BYTES / DW;
            opus::static_for<T::B_VGPR_GROUPS>([&](auto vc) {
                constexpr int v   = vc.value;
                constexpr int imm = v * 2 * T::LDS_BLOCK_BYTES / DW;
                auto x = opus::load<GRP_DW>(s_dw, stage_dw + base + imm + lds_lane_dw);
                opus::set_slice(kv[nt], x, opus::number<v * GRP_DW>{},
                                opus::number<(v + 1) * GRP_DW>{});
            });
        });
    };

    // relu x weight over the lane's 16 heads of one m-tile, as two pk_fma chains (one chain of 8
    // dependent steps stalls); returned unjoined, (even heads, odd heads). One head of m-tile 0
    // goes through the IEEE maximum, so a NaN key's NaN (every head of its column) rides the relu
    // and no per-tile test of the key scales is needed. asm, or -fno-honor-nans lets the compiler
    // pick max_num; a VGPR operand (this kernel is <= 256 VGPRs).
    auto head_sum = [&](const typename Mma::vtype_c& c, const opus::vector_t<float, EC>& wr,
                        auto with_nan_c) {
        constexpr bool WITH_NAN = decltype(with_nan_c)::value;
        opus::vector_t<float, 2> s2[2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        opus::static_for<EC / 2>([&](auto pc) {
            constexpr int p = pc.value;
            opus::vector_t<float, 2> wv{wr[2 * p], wr[2 * p + 1]};
            opus::vector_t<float, 2> rv{relu_num(c[2 * p]), relu_num(c[2 * p + 1])};
            if constexpr (WITH_NAN && p == 0) {
                float x = c[0];
                asm volatile("v_maximum_f32 %0, %0, 0" : "+v"(x));
                rv[0] = x;
            }
            s2[p % 2] = rv * wv + s2[p % 2];
        });
        return s2[0] + s2[1];
    };

    // ---- the first tiles' TDM and this wave's row ----
    constexpr int LEAD = STAGES - 1;   // tiles in flight ahead of the one being computed
    int pf = 0;   // the page id of the next tile to issue, read one tile ahead
    auto issue_first = [&]() {
        opus::static_for<LEAD>([&](auto sc) {
            if (sc.value < tile_count) issue_tile(sc.value, sc, bt_c[sc.value]);
        });
        if (LEAD < tile_count) pf = bt_c[LEAD];
        __builtin_amdgcn_sched_barrier(0);
    };
    // Q is 4 KB a row and every chunk-CTA of a row reads it: one-row CTAs issue it before the
    // TDM, multi-row ones behind it.
    constexpr bool Q_FIRST = T::NUM_WAVES == 1;
    if constexpr (!Q_FIRST) issue_first();

    typename Mma::vtype_a q_a[MT];
    int q_sc[MT];
    opus::vector_t<float, EC> w[MT];
    auto g_q = opus::make_gmem(reinterpret_cast<const D_SCALE*>(
        reinterpret_cast<const D_BYTE*>(k.ptr_q) + (size_t)row_id * T::Q_ROW_BYTES));
    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<T::A_VGPR_GROUPS>([&](auto vc) {
            constexpr int v = vc.value;
            const int head  = mi * T::MMA_M + lane_m0 + (v >= 2 ? 16 : 0);
            const int block = lane_g + (v % 2 ? 2 : 0);
            auto x = opus::load<GRP_DW>(g_q, T::q_byte(head, block) / DW);
            opus::set_slice(q_a[mi], x, opus::number<v * GRP_DW>{},
                            opus::number<(v + 1) * GRP_DW>{});
        });
    });
    auto g_qs = opus::make_gmem(reinterpret_cast<const D_SCALE*>(k.ptr_q_scale) +
                                (size_t)row_id * T::N_HEADS);
    opus::static_for<MT>([&](auto mic) {
        q_sc[mic.value] = opus::load<1>(g_qs, mic.value * T::MMA_M + lane)[0];
    });
    auto g_w = opus::make_gmem(reinterpret_cast<const D_WEIGHT*>(k.ptr_weights) +
                               (size_t)row_id * T::W_ROW_ELEMS);
    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<T::REPT_C>([&](auto hc) {
            constexpr int h = hc.value;
            auto wv = opus::load<PACK_W>(g_w, mi * T::MMA_M + (h ? 16 : 0) + lane_g * PACK_W);
            opus::static_for<PACK_W>([&](auto jc) {
                w[mi][h * PACK_W + jc.value] = (float)wv[jc.value] * weight_scale;
            });
        });
    });
    if constexpr (Q_FIRST) {
        __builtin_amdgcn_sched_barrier(0);
        issue_first();
    }
    // This wave's window, as a window-sized V#: the hardware's bound check masks the stores past
    // it. A masked-off wave has 0 bytes. readfirstlane, not pin_sgpr: uniform, but a scalar load
    // only while the compiler can prove no store clobbers it.
    const int own_end = __builtin_amdgcn_readfirstlane(slot_ok ? k.ptr_row_ends[row_id] : 0);
    const unsigned out_bytes = (unsigned)(own_end > 0 ? own_end : 0) * (unsigned)sizeof(float);
    auto g_out = opus::make_gmem(k.ptr_out + (size_t)row_id * stride_out, out_bytes);

    // The row's q scales hold a NaN (uniform). Tested after every prologue load is out: a branch
    // on a scale per load serialises the round trips.
    const bool q_nan = __builtin_amdgcn_ballot_w32(e8m0x4_has_nan((unsigned)q_sc[0]) ||
                                                   e8m0x4_has_nan((unsigned)q_sc[1])) != 0;

    typename Mma::vtype_b kv_b[NT];
    int kvs[NT / 2];
    // Unrolled by the ring depth so every slot index is a compile-time constant.
    for (int tb = 0; tb < tile_count; tb += STAGES) {
        opus::static_for<STAGES>([&](auto sc) {
            constexpr int S = sc.value;
            const int t = tb + S;
            if (t >= tile_count) return;   // CTA-uniform
            if (chunk_max > 0) set_prio_by_progress(tile_count - t, chunk_max);   // uniform
            // Tile t has landed once at most the tiles issued after it are outstanding; near the
            // end fewer were issued, and the constant would no longer cover t: drain instead.
            if (tile_count - 1 - t >= LEAD - 1) opus::s_wait_tensorcnt<T::TENSORCNT_KEEP>();
            else                                opus::s_wait_tensorcnt<0>();
            cta_sync();
            issue_tile(t + LEAD, opus::number<(S + LEAD) % STAGES>{}, pf);
            if (t + LEAD + 1 < tile_count) pf = bt_c[t + LEAD + 1];
            read_kv(kv_b, kvs, sc);
            __builtin_amdgcn_sched_barrier(0);

            // Accumulator j is m-tile j % MT of n-tile j / MT. An n-tile's head sum is its two
            // m-tiles' (the lane's 32 of the 64 heads; the partner lane has the rest), and the odd
            // n-tile finishes the pair: keys 32u + lane, in out[u].
            float out[NT / 2];
            opus::vector_t<float, 2> hs[MT * NT];
            auto wmma_j = [&](auto jc) {
                constexpr int j = decltype(jc)::value, nt = j / MT, mi = j % MT;
                return mma(q_a[mi], kv_b[nt], q_sc[mi], kvs[nt / 2], opus::number<0>{},
                           opus::number<nt % 2>{});
            };
            auto epilogue_j = [&](auto jc, const typename Mma::vtype_c& c) {
                constexpr int j = decltype(jc)::value, nt = j / MT, mi = j % MT;
                hs[j] = head_sum(c, w[mi], opus::bool_constant<mi == 0>{});
                if constexpr (mi == MT - 1 && nt % 2 == 1) {
                    out[nt / 2] = pair_head_reduce(pair_sum<true>(hs[j - 3] + hs[j - 2]),
                                                   pair_sum<true>(hs[j - 1] + hs[j]));
                }
            };
            // Pipelined by n-tile: n-tile m's two WMMAs, then n-tile m-1's two epilogues, each step
            // fenced so the WMMAs lead and no relu sits right behind the WMMA it reads.
            constexpr int A = NT * MT;
            typename Mma::vtype_c c0 = wmma_j(opus::number<0>{}), c1 = wmma_j(opus::number<1>{});
            opus::static_for<A / 2 - 1>([&](auto mc) {
                constexpr int m = mc.value + 1;
                __builtin_amdgcn_sched_barrier(0);
                typename Mma::vtype_c n0 = wmma_j(opus::number<2 * m>{});
                typename Mma::vtype_c n1 = wmma_j(opus::number<2 * m + 1>{});
                __builtin_amdgcn_sched_barrier(0);
                epilogue_j(opus::number<2 * m - 2>{}, c0);
                epilogue_j(opus::number<2 * m - 1>{}, c1);
                c0 = n0;
                c1 = n1;
            });
            __builtin_amdgcn_sched_barrier(0);
            epilogue_j(opus::number<A - 2>{}, c0);
            epilogue_j(opus::number<A - 1>{}, c1);
            // The keys' NaN rides the relu (the IEEE element); a NaN q row is put back here.
            if (q_nan) {
                opus::static_for<NT / 2>([&](auto uc) {
                    out[uc.value] = std::bit_cast<float>(0x7fc00000u);
                });
            }
            // The stores after the tile's compute: a store's data registers are the accumulators',
            // and a WMMA rewriting them waits for the store to read them out (s_wait_xcnt).
            __builtin_amdgcn_sched_barrier(0);
            const int tile_off = (chunk_start + t) * TK + lane;
            opus::static_for<NT / 2>([&](auto uc) {
                constexpr int u = uc.value;
                g_out.template store<1>(out[u], tile_off + u * 2 * T::MMA_N);
            });
            __builtin_amdgcn_sched_barrier(0);
        });
    }
}

}  // namespace opus_logits::gfx1250
