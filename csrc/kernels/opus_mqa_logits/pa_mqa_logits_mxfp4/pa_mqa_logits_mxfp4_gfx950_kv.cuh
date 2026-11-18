// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 wave64 (MFMA 32x32x64) one-row kernel: one query row a CTA, one wave over each 64-key
// tile, the KV read straight from global into a register double buffer: V4, and V4.1's candidate
// layers (GATHER). Device pass on gfx950 only (pa_mqa_logits_mxfp4_gfx950.cu).
#pragma once
#include "pa_mqa_logits_mxfp4_gfx950_common.cuh"

namespace opus_logits::gfx950 {

// block_tables: entry wave + i, one int a wave. The wave index is the layout's p coordinate, so
// the page-id address is computed per lane (VALU), off the SALU chain the prologue waits on.
template<class T>
__device__ inline auto make_layout_bt(int wave) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::NUM_WAVES>{},  // wave   (p)
        opus::number<1>{});            // one int (y)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(
            opus::number<1>{},
            opus::number<0>{})),
        opus::unfold_p_coord(dim, opus::make_tuple(wave)));
}

// Grid: Table one CTA a record; Free groups + cta_target; Prefill groups x prefill_parts
// (pa_mqa_logits_mxfp4.py sizes them). A group is one row here.
//
// GATHER (V4.1 candidate layers, one row): a table entry is one 8-row page and a tile is 8 of them,
// anywhere in the plane. Lane (g, m) of n-tile nt holds tile key (m / 4) * 8 + nt * 4 + m % 4:
// page m / 4 of the tile in both n-tiles, so a lane reads one page id a tile (a VMEM load, as the
// contiguous walk's) and its n-tiles' keys sit an immediate apart in the page's 128 B a block.
template<class T, int SCHED>
__global__ __launch_bounds__(T::BLOCK_SIZE, MIN_WAVES_PER_EU)
void kv_kernel(mqa_logits_mxfp4_kargs k) {
    using D_DATA   = opus::fp4_t;   // MFMA A/B element (E2M1)
    using D_BYTE   = uint8_t;       // fp4 is addressed in bytes
    using D_WEIGHT = typename T::D_WEIGHT;
    using D_SCALE  = typename T::D_SCALE;
    using D_QSCALE = typename T::D_QSCALE;
    using D_ACC    = typename T::D_ACC;
    using D_OUT    = typename T::D_OUT;

    constexpr int WAVE = T::WAVE_SIZE;
    constexpr int MT   = T::M_TILES;
    constexpr int NT   = T::N_TILES;
    constexpr int KT   = T::K_TILES;
    constexpr int TK   = T::KV_TILE_SIZE;
    constexpr int EC   = T::C_FRAG;

    // Read and pinned ahead of the early exits, so the loads stay one scalar round trip and
    // cannot sink past them.
    const void* p_q        = k.ptr_q;
    const void* p_q_scale  = k.ptr_q_scale;
    const void* p_kv       = k.ptr_kv;
    const void* p_kv_scale = k.ptr_kv_scale;
    const int*  p_bt       = k.ptr_block_tables;
    const void* p_weights  = k.ptr_weights;
    float*      p_out      = k.ptr_out;
    int   stride_out   = k.out_stride;
    float weight_scale = k.weight_scale;
    int   bt_stride    = k.bt_stride;
    mqa_sched_args sa(k);
    pin_sgpr(sa.num_rows);
    pin_sgpr(p_q); pin_sgpr(p_q_scale); pin_sgpr(p_kv); pin_sgpr(p_kv_scale);
    pin_sgpr(p_bt); pin_sgpr(p_weights); pin_sgpr(p_out);
    pin_sgpr(stride_out); pin_sgpr(weight_scale); pin_sgpr(bt_stride);
    [[maybe_unused]] const int* p_out_offsets = k.ptr_out_offsets;
    [[maybe_unused]] unsigned kv_last_row = k.kv_last_row;
    if constexpr (T::FLAT_OUT) pin_sgpr(p_out_offsets);
    if constexpr (T::GATHER) pin_sgpr(kv_last_row);

    const int tid     = opus::thread_id_x();
    const int wave    = tid >> 6;
    const int lane    = tid & (WAVE - 1);
    const int lane_m0 = lane & (T::MMA_N - 1);   // n column / m row within the tile
    const int lane_g  = lane >> 5;               // 32-K block within the k-tile

    // CTA-uniform early return. The window is a prefix [0, local_end).
    mqa_work wk;
    if (!mqa_assign<1, TK, SCHED, T::GATHER>(sa, lane, wk)) return;
    const int row_id      = wk.row_id;
    const int batch_id    = wk.batch_id;
    const int chunk_start = wk.chunk_start;
    const int tile_count  = wk.chunk_count;
    const int local_end   = wk.local_end;

    auto mma = opus::make_tiled_mma<D_DATA, D_DATA, D_ACC>(
        opus::seq<MT, NT, KT>{}, opus::seq<1, 1, 1>{},
        opus::seq<T::MMA_M, T::MMA_N, T::MMA_K>{});
    using Mma = typename decltype(mma)::MMA;
    Mma mma_op;
    static_assert(sizeof(typename Mma::vtype_a) == 2 * T::A_BYTES_PER_LANE,
                  "expected a 256-bit MFMA operand with the fp4 payload in its low half");
    static_assert(Mma::elem_c == EC, "traits C_FRAG must match the MFMA accumulator");

    // The row folded into the base pointers to keep voffsets small.
    const D_BYTE* q_base = reinterpret_cast<const D_BYTE*>(p_q) + (size_t)row_id * T::Q_ROW_BYTES;
    const D_QSCALE* qs_base =
        reinterpret_cast<const D_QSCALE*>(p_q_scale) + (size_t)row_id * WAVE;
    const D_WEIGHT* w_base =
        reinterpret_cast<const D_WEIGHT*>(p_weights) + (size_t)row_id * T::W_ROW_ELEMS;
    const int* bt_base = p_bt + (size_t)batch_id * bt_stride;
    // The out V# spans exactly the window: the hardware's range check is the whole store mask.
    D_OUT* out_base = row_out<T>(p_out, p_out_offsets, stride_out, row_id);
    const unsigned out_bytes = (unsigned)(local_end > 0 ? local_end : 0) * (unsigned)sizeof(D_OUT);

    auto g_q   = opus::make_gmem(q_base);
    auto g_qs  = opus::make_gmem(qs_base);
    auto g_bt  = opus::make_gmem(bt_base);
    auto g_w   = opus::make_gmem(w_base);
    auto g_out = opus::make_gmem(out_base, out_bytes);

    // q_a[mi][kt]: the 16 B payload in the low half of a 256-bit operand, zeroed so the ignored
    // high half is never undefined.
    auto u_q = make_layout_q<T>(lane_m0, lane_g);
    typename Mma::vtype_a q_a[MT][KT]{};
    auto u_qs = make_layout_scale<T>(lane);
    int q_scale;   // the row's scale bytes, zero-extended
    auto u_w = make_layout_w<T>(lane_g);
    opus::vector_t<float, EC> w_pl[MT];
    auto u_bt  = make_layout_bt<T>(wave);
    // A page is [kt][g][nt][m][16 B]; lane L reads K[(kt * K_CHUNKS + g) * 32 : +32] of key
    // nt * MMA_N + m (g = L / 32, m = L % 32), and scale dword L.
    const int lane_kv = lane_g * T::STRIDE_KV_G + lane_m0 * T::STRIDE_KV_M;
    const int key_lane = wave * (NT * T::MMA_N) + lane_m0;   // this lane's key in a tile

    // Lanes L and L^32 hold the same logit; lane group 1 is biased out of bounds and masked. A
    // large negative constant, never a multiple of the window: that could land back inside it.
    const int lane_bias = (lane_g != 0) ? -(chunk_start + tile_count) * TK : 0;

    // Tile t of the walk: its entry's page, or (GATHER) the lane's 8-row page of it. A gathered
    // tile's last ids run past the row's window into slots the caller may have padded with
    // anything: clamped into the pool (unsigned, so a negative id clamps too), each reads a real
    // page, whose keys are then never stored.
    [[maybe_unused]] const int lane_page = lane_m0 / T::GATHER_NT_KEYS;
    [[maybe_unused]] const unsigned last_page = (kv_last_row - (T::PAGE_ROWS - 1)) / T::PAGE_ROWS;
    auto load_page_id = [&](int tile_idx) {
        if constexpr (T::GATHER) {
            const unsigned id = (unsigned)opus::load<1>(
                g_bt, (chunk_start + tile_idx) * T::TILE_PAGES + lane_page)[0];
            return (int)(id < last_page ? id : last_page);
        } else {
            return opus::load<1>(g_bt, u_bt + (chunk_start + tile_idx))[0];
        }
    };

    using kv_nt_t = typename Mma::vtype_b;
    using sfrag   = opus::vector_t<float, MT * EC>;
    using kvs_t   = std::conditional_t<T::GATHER, gather_scales, int>;

    auto clamp_tile = [&](int t) { return t < tile_count ? t : tile_count - 1; };
    // GATHER: lane (g, m)'s bytes of its page p -- group p / 8, rows (p % 8) * 8 .. + 7 -- and
    // scale dwords. Key (p % 8) * 8 + nt * 4 + m % 4 of the group: block g at
    // group + kt * 2048 + g * 1024 + key * 16; its scales in dword g * 32 + key % 32 at byte
    // kt * 2 + key / 32 (the selector moves that to kt * 2 + nt, from dword lo or hi).
    [[maybe_unused]] const int lane_kv_g =
        lane_g * T::STRIDE_KV_G + (lane_m0 % T::GATHER_NT_KEYS) * T::STRIDE_KV_M;
    [[maybe_unused]] const int lane_kvs_g = lane_g * T::MMA_N + lane_m0 % T::GATHER_NT_KEYS;
    // Each lane its own page, so no wave-uniform base for a per-page V#: 64-bit addresses
    // (global_load_as).
    using kv_grp_t = opus::vector_t<D_BYTE, T::KV_GRP_BYTES>;
    auto issue_gather = [&](kv_nt_t (&kv)[NT][KT], gather_scales& kvs, int page_id) {
        const auto kv_base  = reinterpret_cast<gbyte_ptr>(reinterpret_cast<__UINTPTR_TYPE__>(p_kv));
        const auto kvs_base =
            reinterpret_cast<gbyte_ptr>(reinterpret_cast<__UINTPTR_TYPE__>(p_kv_scale));
        const int grp = page_id / T::TILE_PAGES, sub = page_id % T::TILE_PAGES;
        const gbyte_ptr kv_addr = kv_base + (size_t)(unsigned)grp * T::KV_PAGE_BYTES +
                                  (unsigned)(sub * T::PAGE_ROWS * T::STRIDE_KV_M + lane_kv_g);
        const gbyte_ptr kvs_addr =
            kvs_base + (size_t)(unsigned)grp * T::KVS_PAGE_BYTES +
            (unsigned)((sub % (T::TILE_PAGES / 2)) * T::PAGE_ROWS + lane_kvs_g) * sizeof(D_SCALE);
        kvs.lo  = global_load_as<D_SCALE>(kvs_addr);
        kvs.hi  = global_load_as<D_SCALE>(kvs_addr + T::GATHER_NT_KEYS * sizeof(D_SCALE));
        // byte kt * 2 + nt <- byte kt * 2 + half of lo (nt 0: selector 4..7) or hi (nt 1: 0..3)
        kvs.sel = 0x02060004 + (sub / (T::TILE_PAGES / 2)) * 0x01010101;
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            opus::static_for<KT>([&](auto ktc) {
                constexpr int kt = ktc.value;
                auto v = global_load_as<kv_grp_t>(
                    kv_addr + (kt * T::STRIDE_KV_KTILE + nt * T::GATHER_NT_KEYS * T::STRIDE_KV_M));
                opus::set_slice(kv[nt][kt], v, opus::number<0>{}, opus::number<T::KV_GRP_BYTES>{});
            });
        });
    };
    // NT * KT independent 16 B loads, each covering two 512 B runs.
    auto issue_kv = [&](kv_nt_t (&kv)[NT][KT], kvs_t& kvs, int page_id) {
        if constexpr (T::GATHER) {
            issue_gather(kv, kvs, page_id);
        } else {
            // A V# a page, off the page id made scalar (one entry a wave: wave-uniform), its base
            // 64-bit scalar arithmetic: the offsets stay inside the page.
            const unsigned pg = __builtin_amdgcn_readfirstlane(page_id);
            auto g_kvs = opus::make_gmem(reinterpret_cast<const D_SCALE*>(p_kv_scale) +
                                         (size_t)pg * (T::KVS_PAGE_BYTES / sizeof(D_SCALE)));
            auto g_kv  = opus::make_gmem(reinterpret_cast<const D_BYTE*>(p_kv) +
                                         (size_t)pg * T::KV_PAGE_BYTES);
            kvs = opus::load<1>(g_kvs, lane)[0];
            opus::static_for<NT>([&](auto ntc) {
                constexpr int nt = ntc.value;
                opus::static_for<KT>([&](auto ktc) {
                    constexpr int kt = ktc.value;
                    auto v = opus::load<T::KV_GRP_BYTES>(
                        g_kv, lane_kv + kt * T::STRIDE_KV_KTILE + nt * T::STRIDE_KV_NTILE);
                    opus::set_slice(kv[nt][kt], v, opus::number<0>{},
                                    opus::number<T::KV_GRP_BYTES>{});
                });
            });
        }
    };

    // One phase: refill the buffer tile t freed with tile t + 2, then tile t + 1's GEMMs, tile t's
    // relu and head reduction. The next page id is read ahead of the refill: vmcnt retires in
    // order, so a wait for an id read behind the tile's loads would wait for them all.
    auto compute_phase = [&](kv_nt_t (&kv_c)[NT][KT], kvs_t& kvs_c, sfrag (&acc_c)[NT],
                             kv_nt_t (&kv_p)[NT][KT], kvs_t& kvs_p, sfrag (&acc_p)[NT],
                             int& pg_c, int pf_tile, opus::vector_t<float, NT>& out) {
        __builtin_amdgcn_sched_barrier(0);
        const int pg_n = load_page_id(clamp_tile(pf_tile + 2));
        issue_kv(kv_c, kvs_c, pg_c);
        pg_c = pg_n;
        __builtin_amdgcn_sched_barrier(0);
        compute_tile<T>(mma_op, q_a, q_scale, w_pl, acc_c, acc_p, kv_p, kv_scale_word(kvs_p), out);
    };
    // The lane's key of n-tile 0 in a tile (n-tile nt: + nt * the n-tile's step).
    constexpr int KEY_NT_STEP = T::GATHER ? T::GATHER_NT_KEYS : T::MMA_N;
    const int key_col =
        T::GATHER ? lane_page * T::PAGE_ROWS + lane_m0 % T::GATHER_NT_KEYS : key_lane;
    // Lanes L and L^32 hold the same logit after the head reduction; lane group 1 is biased out of
    // bounds. Fenced so the stores stay out of the MFMA groups' interleave.
    auto store = [&](opus::vector_t<float, NT>& out, int t) {
        __builtin_amdgcn_sched_barrier(0);
        const int tile_off = (chunk_start + t) * TK + lane_bias;
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            g_out.template store<1>(out[nt], tile_off + nt * KEY_NT_STEP + key_col);
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    // Zero-initialized: fp4 fills only the low 16 B of each 256-bit operand.
    kv_nt_t kvA[NT][KT]{}, kvB[NT][KT]{};
    kvs_t   kvsA, kvsB;
    sfrag   accA[NT], accB[NT];
    int     pgA, pgB;
    opus::vector_t<float, NT> outA, outB;

    // Prologue: Q, q_scale and weights straight from global, then KV tiles 0 and 1.
    load_q<T>(g_q, u_q, q_a);
    q_scale = opus::load<1>(g_qs, u_qs)[0];
    load_weights<T>(g_w, u_w, w_pl, weight_scale);

    // Whole-tile lead: tile 0 is fully computed into accA so phase 0 can reload kvA at once;
    // tile 1 is staged in kvB for that phase's GEMMs.
    issue_kv(kvA, kvsA, load_page_id(clamp_tile(0)));
    issue_kv(kvB, kvsB, load_page_id(clamp_tile(1)));
    pgA = load_page_id(clamp_tile(2));
    gemm_ntile<T, 0>(mma_op, q_a, q_scale, accA[0], kvA[0], kv_scale_word(kvsA));
    gemm_ntile<T, 1>(mma_op, q_a, q_scale, accA[1], kvA[1], kv_scale_word(kvsA));
    pgB = load_page_id(clamp_tile(3));
    __builtin_amdgcn_sched_barrier(0);

    compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, pgA, 2, outA);
    __builtin_amdgcn_sched_barrier(0);

    // Each phase stores its own result: stores lagged a phase tighten the loop-head s_waitcnt
    // (their WAR with the next MFMA) beyond what the KV needs.
    store(outA, 0);
    if (tile_count >= 2) {
        compute_phase(kvB, kvsB, accB, kvA, kvsA, accA, pgB, 3, outB);
        store(outB, 1);
    }

    int t = 2;
    for (; t + 1 < tile_count; t += 2) {
        compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, pgA, t + 2, outA);
        store(outA, t);
        compute_phase(kvB, kvsB, accB, kvA, kvsA, accA, pgB, t + 3, outB);
        store(outB, t + 1);
    }
    if (t < tile_count) {
        compute_phase(kvA, kvsA, accA, kvB, kvsB, accB, pgA, t + 2, outA);
        store(outA, t);
    }
}

}  // namespace opus_logits::gfx950
